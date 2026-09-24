"""
ITRS-to-GCRS (ECEF-to-ECI/J2000) Frame Transformation — Implementation & Verification
=====================================================================================

This module implements the ITRS→GCRS transformation consistent with
IERS Conventions 2010 (Technical Note 36, Chapter 5) and IAU 2006/2000A
resolutions, using the CIO-based formulation:

    [GCRS] = Q(t) · R(t) · W(t) · [ITRS]

where:
    W(t) = polar motion matrix      (ITRS → TIRS)
    R(t) = Earth rotation matrix    (TIRS → CIRS)
    Q(t) = celestial pole matrix    (CIRS → GCRS)

The SOFA/ERFA convention defines the celestial-to-terrestrial (GCRS→ITRS)
matrix as:

    [ITRS] = W · R₃(ERA) · Q_c2i · [GCRS]

so the terrestrial-to-celestial (ITRS→GCRS) matrix is the transpose:

    [GCRS] = Q_c2iᵀ · R₃(ERA)ᵀ · Wᵀ · [ITRS]

This implementation builds each component independently from ERFA primitives,
then assembles the full matrix, and verifies against:
  1. Official ERFA test vectors (t_erfa_c.c)
  2. IAU 2006 reference values (Capitaine et al. 2006, A&A 450, 855)
  3. Direct comparison with erfa.c2t06a()

Common pitfalls are explicitly tested.

References:
  - IERS Conventions 2010, Petit & Luzum (eds.), IERS Technical Note 36, Ch. 5
  - Capitaine et al. (2006), "Precession-nutation procedures consistent with
    IAU 2006 resolutions", A&A 450, 855–876
  - Wallace, P. (2007), "SOFA support for the IAU 2006 precession model"
  - SOFA Tools for Earth Attitude (www.iausofa.org)
  - ERFA library (liberfa/erfa on GitHub), test file t_erfa_c.c
"""
#%%
import numpy as np
import erfa

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# IAU 2006 constants
DAS2R = np.pi / (180.0 * 3600.0)    # arcseconds to radians
DJM0 = 2400000.5                    # MJD Julian date offset
DJM00 = 51544.5                     # J2000.0 as MJD
DJ00 = 2451545.0                    # J2000.0 as JD

# GPS epoch: JD 2444244.5 (1980-01-06T00:00:00 UTC)
GPS_JD0 = 2444244.5

# TT - TAI = 32.184 s  (IAU fixed)
TT_MINUS_TAI = 32.184

# TAI - UTC = 19 (as of 1972-07-01), then incremental leap seconds
# GPS - TAI = -19 s  (GPS is ahead of UTC by leap_seconds + 19)

# ---------------------------------------------------------------------------
# Time conversion utilities
# ---------------------------------------------------------------------------

    # Known leap seconds: (effective_from_utc_jd, tai_minus_utc)
# TAI - UTC values after each leap second insertion
LEAP_SECONDS_TABLE = [
    # (effective_from_utc_jd, tai_minus_utc)
    (2441317.5, 10),   # 1972-01-01
    (2441499.5, 11),   # 1972-07-01
    (2441683.5, 12),   # 1973-01-01
    (2442048.5, 13),   # 1974-01-01
    (2442413.5, 14),   # 1975-01-01
    (2442778.5, 15),   # 1976-01-01
    (2443144.5, 16),   # 1977-01-01
    (2443509.5, 17),   # 1978-01-01
    (2443874.5, 18),   # 1979-01-01
    (2444239.5, 19),   # 1980-01-01
    (2444786.5, 20),   # 1981-07-01
    (2445151.5, 21),   # 1982-07-01
    (2445516.5, 22),   # 1983-07-01
    (2446247.5, 23),   # 1985-07-01
    (2447161.5, 24),   # 1988-01-01
    (2447892.5, 25),   # 1990-01-01
    (2448257.5, 26),   # 1991-01-01
    (2448804.5, 27),   # 1992-07-01
    (2449169.5, 28),   # 1993-07-01
    (2449534.5, 29),   # 1994-07-01
    (2450083.5, 30),   # 1996-01-01
    (2450630.5, 31),   # 1997-07-01
    (2451179.5, 32),   # 1999-01-01
    (2453736.5, 33),   # 2006-01-01
    (2454832.5, 34),   # 2009-01-01
    (2456109.5, 35),   # 2012-07-01
    (2457204.5, 36),   # 2015-07-01
    (2457754.5, 37),   # 2017-01-01
]


def gps_to_utc(gps_seconds):
    """Convert GPS time (seconds since GPS epoch) to UTC JD.

    GPS time = TAI - 19s. UTC = TAI - leap_seconds.
    GPS is AHEAD of UTC by (leap_seconds - 19) seconds.
    So UTC = GPS - (leap_seconds - 19).
    """
    # GPS epoch in JD
    gps_jd = GPS_JD0 + gps_seconds / 86400.0

    # Find the leap seconds applicable
    tai_minus_utc = 19  # default
    for eff_jd, ls in LEAP_SECONDS_TABLE:
        if gps_jd >= eff_jd:
            tai_minus_utc = ls
        else:
            break

    # GPS is ahead of UTC by (tai_minus_utc - 19) seconds
    # UTC = GPS - (tai_minus_utc - 19)
    gps_minus_utc = tai_minus_utc - 19
    utc_jd = gps_jd - gps_minus_utc / 86400.0
    return utc_jd


def utc_to_tt(utc_jd):
    """Convert UTC JD to TT JD (2-part).

    Returns (jd1, jd2) as 2-part Julian Date for maximum precision.
    """
    # TAI - UTC from leap second table
    tai_minus_utc = 19
    for eff_jd, ls in LEAP_SECONDS_TABLE:
        if utc_jd >= eff_jd:
            tai_minus_utc = ls
        else:
            break

    # TT = UTC + (TAI-UTC) + 32.184s
    delta_seconds = tai_minus_utc + TT_MINUS_TAI
    tt_jd = utc_jd + delta_seconds / 86400.0

    # Split into 2-part JD for maximum precision
    jd1 = np.floor(tt_jd)
    jd2 = tt_jd - jd1
    # Use MJD splitting for better precision
    jd1 = DJM0
    jd2 = tt_jd - DJM0
    return jd1, jd2


def gps_to_tt(gps_seconds):
    """Convert GPS seconds to TT as 2-part JD."""
    utc_jd = gps_to_utc(gps_seconds)
    return utc_to_tt(utc_jd)


# ---------------------------------------------------------------------------
# Core transformation: ITRS → GCRS
# ---------------------------------------------------------------------------

def rz(angle):
    """Rotation matrix about z-axis by 'angle' (radians), right-handed.

    R₃(θ) = | cos θ  sin θ  0 |
            |-sin θ  cos θ  0 |
            |  0      0     1 |

    Note: This is the SOFA convention where R₃(θ) rotates vectors.
    """
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([
        [c,  s, 0],
        [-s, c, 0],
        [0,  0, 1],
    ])


def itrs_to_gcrs(tt_jd1, tt_jd2, ut1_jd1, ut1_jd2, xp, yp,
                 dx=0.0, dy=0.0):
    """Transform coordinates from ITRS (ECEF) to GCRS (ECI/J2000).

    Uses the CIO-based formulation consistent with IERS Conventions 2010:

        [GCRS] = Q(t) · R(t) · W(t) · [ITRS]

    Parameters
    ----------
    tt_jd1, tt_jd2 : float
        TT as 2-part Julian Date (for precession-nutation and CIO locator)
    ut1_jd1, ut1_jd2 : float
        UT1 as 2-part Julian Date (for Earth rotation angle)
    xp, yp : float
        Polar motion coordinates in radians.
        xp: coordinate of the CIP w.r.t. ITRS, measured along the
            meridian 0° (Greenwich)
        yp: coordinate of the CIP w.r.t. ITRS, measured along the
            meridian 90°W
    dx, dy : float, optional
        Celestial pole offsets (dX, dY) in radians, added to the IAU
        2006/2000A CIP coordinates.

    Returns
    -------
    rc2t_transpose : ndarray, shape (3, 3)
        The ITRS→GCRS transformation matrix.
        Also returns a dict with intermediate quantities.

    Notes
    -----
    The SOFA convention defines the GCRS→ITRS matrix as:

        [ITRS] = W · R₃(ERA) · Q_c2i · [GCRS]

    The ITRS→GCRS matrix is therefore:

        [GCRS] = Q_c2iᵀ · R₃(ERA)ᵀ · Wᵀ · [ITRS]

    Equivalently, Q = Q_c2iᵀ, R = R₃(-ERA), W = W_matrixᵀ, and
    the product Q·R·W gives the ITRS→GCRS matrix.
    """
    # ------------------------------------------------------------------
    # Step 1: Celestial-to-intermediate matrix Q_c2i (GCRS → CIRS)
    #         Uses IAU 2006 precession + IAU 2000A nutation
    # ------------------------------------------------------------------
    # CIP coordinates X, Y and CIO locator s (IAU 2006/2000A)
    x, y, s = erfa.xys06a(tt_jd1, tt_jd2)

    # Apply celestial pole offsets if provided
    x = x + dx
    y = y + dy

    # GCRS-to-CIRS matrix from X, Y, s
    # This is Q_c2i in the SOFA convention
    rc2i = erfa.c2ixys(x, y, s)

    # ------------------------------------------------------------------
    # Step 2: Earth Rotation Angle (ERA) — TIRS rotation
    #         R = R₃(ERA) rotates CIRS → TIRS
    # ------------------------------------------------------------------
    era = erfa.era00(ut1_jd1, ut1_jd2)

    # ------------------------------------------------------------------
    # Step 3: Polar motion matrix W (TIRS → ITRS)
    #         Uses TIO locator s' (sp)
    # ------------------------------------------------------------------
    sp = erfa.sp00(tt_jd1, tt_jd2)   # TIO locator s'
    rpom = erfa.pom00(xp, yp, sp)   # TIRS-to-ITRS matrix

    # ------------------------------------------------------------------
    # Step 4: Assemble the ITRS→GCRS matrix
    # ------------------------------------------------------------------
    # SOFA C2T = W · R₃(ERA) · Q_c2i   (GCRS→ITRS)
    # ITRS→GCRS = C2Tᵀ = Q_c2iᵀ · R₃(ERA)ᵀ · Wᵀ
    #
    # In the IERS Conventions notation:
    #   Q = Q_c2iᵀ  (CIRS→GCRS)
    #   R = R₃(-ERA) (TIRS→CIRS)
    #   W = rpomᵀ   (ITRS→TIRS)
    #   ITRS→GCRS = Q · R · W

    # Method 1: Transpose of the C2T matrix
    #------------ Forward: GCRS -> ITRS---------------------------
    # GCRS --rc2i--> CIRS --rz(ERA)--> TIRS --rpom--> ITRS
    #--------------------------------------------------------------
    # C2T = rpom @ rz(era) @ rc2i
    #------------ Reverse: ITRS -> GCRS---------------------------
    # ITRS --rpom.T--> TIRS --rz(-ERA)--> CIRS --rc2i.T--> GCRS
    #--------------------------------------------------------------
    # T2C = (rpom @ rz(era) @ rc2i).T

    # Method 2: Direct assembly (Q*R*W)
    # In order of operation 1st W, then R then Q 
    W = rpom.T          # ITRS → TIRS
    R = rz(-era)        # TIRS → CIRS  (note: negative ERA!)
    Q = rc2i.T          # CIRS → GCRS
    

    t2c = Q @ R @ W

    info = {
        'x': x, 'y': y, 's': s,
        'era': era,
        'sp': sp,
        'rc2i': rc2i,
        'rpom': rpom,
        'Q': Q, 'R': R, 'W': W,
        't2c': t2c,
        'c2t_expected': rpom @ rz(era) @ rc2i,
    }

    return t2c, info


def itrs_to_gcrs_vector(r_itrs, tt_jd1, tt_jd2, ut1_jd1, ut1_jd2,
                        xp, yp, dx=0.0, dy=0.0):
    """Transform a position vector from ITRS to GCRS.

    Parameters
    ----------
    r_itrs : array_like, shape (3,)
        Position vector in ITRS (meters)
    Other parameters same as itrs_to_gcrs().

    Returns
    -------
    r_gcrs : ndarray, shape (3,)
        Position vector in GCRS (meters)
    """
    t2c, _ = itrs_to_gcrs(tt_jd1, tt_jd2, ut1_jd1, ut1_jd2, xp, yp, dx, dy)
    r_itrs = np.asarray(r_itrs, dtype=float)
    return t2c @ r_itrs


# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------

class TestResults:
    """Simple test result collector."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.details = []

    def check(self, name, actual, expected, tol=1e-12, unit=""):
        diff = abs(actual - expected)
        status = "PASS" if diff <= tol else "FAIL"
        if status == "PASS":
            self.passed += 1
        else:
            self.failed += 1
        self.details.append(f"  [{status}] {name}: got {actual:.18e}, "
                            f"expected {expected:.18e}, diff={diff:.2e} {unit}")
        return status == "PASS"

    def check_matrix(self, name, actual, expected, tol=1e-12):
        diff = np.max(np.abs(np.array(actual) - np.array(expected)))
        status = "PASS" if diff <= tol else "FAIL"
        if status == "PASS":
            self.passed += 1
        else:
            self.failed += 1
        self.details.append(f"  [{status}] {name}: max diff={diff:.2e}")
        return status == "PASS"

    def check_true(self, name, condition, detail=""):
        status = "PASS" if condition else "FAIL"
        if status == "PASS":
            self.passed += 1
        else:
            self.failed += 1
        self.details.append(f"  [{status}] {name}: {detail}")
        return status == "PASS"

    def summary(self):
        total = self.passed + self.failed
        return (f"\n{'='*70}\n"
                f"TEST SUMMARY: {self.passed}/{total} passed, "
                f"{self.failed} failed\n{'='*70}\n")


# ===========================================================================
# Test Case 1: ERFA Official Test Vectors (from t_erfa_c.c)
# ===========================================================================

def test_erfa_era00(tr):
    """Test ERA (Earth Rotation Angle) against ERFA test vector.

    From t_erfa_c.c, test_era00():
      Input:  UT1 JD = (2400000.5, 54388.0)
      Expected: era = 0.4022837240028158102
    """
    print("\n--- Test 1: ERA (Earth Rotation Angle) — ERFA test vector ---")

    ut1a, ut1b = 2400000.5, 54388.0
    expected = 0.4022837240028158102

    # Our implementation (uses erfa.era00 internally)
    era = erfa.era00(ut1a, ut1b)
    tr.check("ERA value", era, expected, tol=1e-14)

    # Verify it's in [0, 2π)
    tr.check_true("ERA in valid range", 0 <= era < 2*np.pi,
                 f"era={era}")

    # Convert to degrees for readability
    era_deg = np.degrees(era)
    print(f"  ERA = {era_deg:.6f}° ({era_deg * 4 / 60:.4f} minutes of rotation)")


def test_erfa_sp00(tr):
    """Test TIO locator s' against ERFA test vector.

    From t_erfa_c.c, test_sp00():
      Input:  TT JD = (2400000.5, 52541.0)
      Expected: sp = -0.6216698469981019309e-11
    """
    print("\n--- Test 2: TIO locator s' (sp) — ERFA test vector ---")

    tt1, tt2 = 2400000.5, 52541.0
    expected = -0.6216698469981019309e-11

    sp = erfa.sp00(tt1, tt2)
    tr.check("TIO locator s'", sp, expected, tol=1e-14)

    print(f"  s' = {sp:.6e} rad ({sp/DAS2R:.6f} arcsec)")


def test_erfa_xys06a(tr):
    """Test CIP X, Y and CIO locator s against ERFA test vector.

    From t_erfa_c.c, test_xys06a():
      Input:  TT JD = (2400000.5, 53736.0)
      Expected: X = 0.5791308482835292617e-3
                Y = 0.4020580099454020310e-4
                s = -0.1220032294164579896e-7
    """
    print("\n--- Test 3: CIP (X, Y) and CIO locator s — ERFA test vector ---")

    tt1, tt2 = 2400000.5, 53736.0

    x, y, s = erfa.xys06a(tt1, tt2)

    tr.check("CIP X", x, 0.5791308482835292617e-3, tol=1e-14)
    tr.check("CIP Y", y, 0.4020580099454020310e-4, tol=1e-15)
    tr.check("CIO locator s", s, -0.1220032294164579896e-7, tol=1e-18)

    print(f"  X = {x:.15e} rad ({np.degrees(x)*3600:.6f} arcsec)")
    print(f"  Y = {y:.15e} rad ({np.degrees(y)*3600:.6f} arcsec)")
    print(f"  s = {s:.15e} rad ({np.degrees(s)*3600:.9f} arcsec)")

    # Verify Z = sqrt(1 - X² - Y²) ≈ 1
    z = np.sqrt(1.0 - x**2 - y**2)
    tr.check_true("CIP Z ≈ 1", abs(z - 1.0) < 1e-6, f"Z={z}")


def test_erfa_s06a(tr):
    """Test CIO locator s (full IAU 2006/2000A) against ERFA test vector.

    From t_erfa_c.c, test_s06a():
      Input:  TT JD = (2400000.5, 52541.0)
      Expected: s = -0.1340680437291812383e-7
    """
    print("\n--- Test 4: CIO locator s (s06a) — ERFA test vector ---")

    tt1, tt2 = 2400000.5, 52541.0
    expected = -0.1340680437291812383e-7

    s = erfa.s06a(tt1, tt2)
    tr.check("CIO locator s (s06a)", s, expected, tol=1e-18)

    print(f"  s = {s:.15e} rad ({np.degrees(s)*3600:.9f} arcsec)")


def test_erfa_pom00(tr):
    """Test polar motion matrix against ERFA test vector.

    From t_erfa_c.c, test_pom00():
      Input:  xp = 2.55060238e-7, yp = 1.860359247e-6,
             sp = -0.1367174580728891460e-10
      Expected: rpom matrix (9 elements)
    """
    print("\n--- Test 5: Polar motion matrix (pom00) — ERFA test vector ---")

    xp = 2.55060238e-7
    yp = 1.860359247e-6
    sp = -0.1367174580728891460e-10

    expected = np.array([
        [0.9999999999999674721,  -0.1367174580728846989e-10,  0.2550602379999972345e-6],
        [0.1414624947957029801e-10, 0.9999999999982695317,    -0.1860359246998866389e-5],
        [-0.2550602379741215021e-6, 0.1860359247002414021e-5,   0.9999999999982370039],
    ])

    rpom = erfa.pom00(xp, yp, sp)
    tr.check_matrix("Polar motion matrix rpom", rpom, expected, tol=1e-12)

    # Verify it's a proper rotation (det ≈ 1, orthogonal)
    det = np.linalg.det(rpom)
    orth_err = np.max(np.abs(rpom @ rpom.T - np.eye(3)))
    tr.check_true("rpom determinant ≈ 1", abs(det - 1.0) < 1e-10, f"det={det}")
    tr.check_true("rpom orthogonal", orth_err < 1e-10, f"||RR^T - I||={orth_err}")

    print(f"  xp = {xp:.6e} rad ({np.degrees(xp)*3600:.6f} arcsec)")
    print(f"  yp = {yp:.6e} rad ({np.degrees(yp)*3600:.6f} arcsec)")


def test_erfa_c2t06a(tr):
    """Test the full celestial-to-terrestrial matrix against ERFA test vector.

    From t_erfa_c.c, test_c2t06a():
      Input:  TT  JD = (2400000.5, 53736.0)
              UT1 JD = (2400000.5, 53736.0)
              xp = 2.55060238e-7, yp = 1.860359247e-6
      Expected: rc2t matrix (9 elements)

    This tests the GCRS→ITRS matrix. The ITRS→GCRS matrix is its transpose.
    """
    print("\n--- Test 6: Full C2T matrix (c2t06a) — ERFA test vector ---")

    tta, ttb = 2400000.5, 53736.0
    uta, utb = 2400000.5, 53736.0
    xp = 2.55060238e-7
    yp = 1.860359247e-6

    expected_c2t = np.array([
        [-0.1810332128305897282,  0.9834769806938592296,  0.6555550962998436505e-4],
        [-0.9834768134136214897, -0.1810332203649130832,  0.5749800844905594110e-3],
        [0.5773474024748545878e-3, 0.3961816829632690581e-4, 0.9999998325501747785],
    ])

    # Get the ERFA reference
    rc2t_erfa = erfa.c2t06a(tta, ttb, uta, utb, xp, yp)
    tr.check_matrix("ERFA c2t06a matrix", rc2t_erfa, expected_c2t, tol=1e-12)

    # Now test our from-scratch implementation
    # Our function returns the ITRS→GCRS matrix (transpose of c2t)
    t2c, info = itrs_to_gcrs(tta, ttb, uta, utb, xp, yp)

    # ITRS→GCRS = (GCRS→ITRS)^T
    expected_t2c = expected_c2t.T
    tr.check_matrix("Our ITRS→GCRS matrix", t2c, expected_t2c, tol=1e-12)

    # Cross-check: our matrix should match erfa.c2t06a(...).T
    tr.check_matrix("Our matrix vs ERFA c2t06a().T",
                    t2c, rc2t_erfa.T, tol=1e-14)

    # Verify it's a proper rotation
    det = np.linalg.det(t2c)
    orth_err = np.max(np.abs(t2c @ t2c.T - np.eye(3)))
    tr.check_true("T2C determinant ≈ 1", abs(det - 1.0) < 1e-10, f"det={det}")
    tr.check_true("T2C orthogonal", orth_err < 1e-10, f"||MM^T - I||={orth_err}")

    # Test vector round-trip: ITRS→GCRS→ITRS should recover original
    r_itrs = np.array([7000000.0, 0.0, 0.0])
    r_gcrs = t2c @ r_itrs
    r_itrs_back = rc2t_erfa @ r_gcrs
    roundtrip_err = np.max(np.abs(r_itrs_back - r_itrs))
    tr.check_true("Round-trip ITRS→GCRS→ITRS", roundtrip_err < 1e-6,
                 f"error={roundtrip_err:.2e} m")


def test_erfa_c2tcio(tr):
    """Test the CIO-based matrix assembly against ERFA test vector.

    From t_erfa_c.c, test_c2tcio():
      Uses pre-computed rc2i, era, and rpom matrices.
    """
    print("\n--- Test 7: CIO-based assembly (c2tcio) — ERFA test vector ---")

    # Pre-computed inputs from the ERFA test
    rc2i = np.array([
        [0.9999998323037164738,   0.5581526271714303683e-9,  -0.5791308477073443903e-3],
        [-0.2384266227524722273e-7, 0.9999999991917404296,  -0.4020594955030704125e-4],
        [0.5791308472168153320e-3,   0.4020595661593994396e-4, 0.9999998314954572365],
    ])

    era = 1.75283325530307

    rpom = np.array([
        [0.9999999999999674705,  -0.1367174580728847031e-10,  0.2550602379999972723e-6],
        [0.1414624947957029721e-10, 0.9999999999982694954,    -0.1860359246998866338e-5],
        [-0.2550602379741215275e-6, 0.1860359247002413923e-5,   0.9999999999982369658],
    ])

    expected_c2t = np.array([
        [-0.1810332128307110439,  0.9834769806938470149,  0.6555535638685466874e-4],
        [-0.9834768134135996657, -0.1810332203649448367,  0.5749801116141106528e-3],
        [0.5773474014081407076e-3, 0.3961832391772658944e-4, 0.9999998325501691969],
    ])

    # ERFA assembly
    rc2t = erfa.c2tcio(rc2i, era, rpom)
    tr.check_matrix("CIO-based C2T assembly", rc2t, expected_c2t, tol=1e-12)

    # Our manual assembly: C2T = W @ Rz(ERA) @ Q_c2i
    c2t_manual = rpom @ rz(era) @ rc2i
    tr.check_matrix("Manual W@Rz(ERA)@Q_c2i assembly",
                    c2t_manual, expected_c2t, tol=1e-12)

    # And ITRS→GCRS = C2T^T
    t2c_manual = c2t_manual.T
    t2c_expected = expected_c2t.T
    tr.check_matrix("Manual T2C = (W@R@Q)^T", t2c_manual, t2c_expected, tol=1e-12)


# ===========================================================================
# Test Case 2: IAU 2006 Reference Values (Capitaine et al. 2006)
# ===========================================================================

def test_iau2006_reference_values(tr):
    """Test against canonical IAU 2006 reference values.

    From: Capitaine, Wallace & Chapront (2006), A&A 450, 855–876
    "Precession-nutation procedures consistent with IAU 2006 resolutions"

    Test date: 2006 January 15, 21h 24m 37.5s UTC
    TT JD  = 2400000.5 + 53750.892855138888889
    UT1 JD = 2400000.5 + 53750.892104561342593
    (DUT1 = +0.3341s)
    """
    print("\n--- Test 8: IAU 2006 reference values (Capitaine et al. 2006) ---")

    # Test date constants
    tt1, tt2 = 2400000.5, 53750.892855138888889
    ut1, ut2 = 2400000.5, 53750.892104561342593

    # --- CIP coordinates X, Y ---
    x, y, s = erfa.xys06a(tt1, tt2)
    tr.check("CIP X (IAU2006)", x, 0.000584859819249, tol=1e-12)
    tr.check("CIP Y (IAU2006)", y, 0.000041535242468, tol=1e-12)

    # --- CIO locator s ---
    # Reference: -0.002571986 arcsec
    s_expected = -0.002571986 * DAS2R
    tr.check("CIO locator s (IAU2006)", s, s_expected, tol=1e-12)

    # --- Earth Rotation Angle ---
    era = erfa.era00(ut1, ut2)
    # Reference: 76.265431053522°
    era_expected = np.radians(76.265431053522)
    tr.check("ERA (IAU2006)", era, era_expected, tol=1e-9)

    # --- Equation of the Origins ---
    eo = erfa.eo06a(tt1, tt2)
    # Reference: -277.646996035 arcsec
    eo_expected = -277.646996035 * DAS2R
    tr.check("Equation of Origins (IAU2006)", eo, eo_expected, tol=1e-9)

    # --- Nutation (IAU 2006/2000A) ---
    # nut06a returns (dpsi, deps) as a tuple
    dpsi, deps = erfa.nut06a(tt1, tt2)
    # Reference: dpsi = -1.071332969 arcsec, deps = +8.656841020 arcsec
    dpsi_expected = -1.071332969 * DAS2R
    deps_expected = 8.656841020 * DAS2R
    tr.check("Nutation dpsi (IAU2006)", dpsi, dpsi_expected, tol=1e-9)
    tr.check("Nutation deps (IAU2006)", deps, deps_expected, tol=1e-9)

    # --- Mean obliquity of date ---
    eps_a = erfa.obl06(tt1, tt2)
    # Reference: 84378.576696215 arcsec
    eps_expected = 84378.576696215 * DAS2R
    tr.check("Mean obliquity (IAU2006)", eps_a, eps_expected, tol=1e-9)

    # --- Full ITRS→GCRS matrix with no polar motion ---
    # Use small xp, yp to test the core precession-nutation + ERA
    xp, yp = 0.0, 0.0
    t2c, info = itrs_to_gcrs(tt1, tt2, ut1, ut2, xp, yp)

    # Cross-check against ERFA c2t06a
    rc2t = erfa.c2t06a(tt1, tt2, ut1, ut2, xp, yp)
    tr.check_matrix("ITRS→GCRS vs c2t06a().T (IAU2006 date)",
                    t2c, rc2t.T, tol=1e-14)

    print(f"  X = {x:.15e} rad ({x/DAS2R:.6f} arcsec)")
    print(f"  Y = {y:.15e} rad ({y/DAS2R:.6f} arcsec)")
    print(f"  s = {s:.15e} rad ({s/DAS2R:.6f} arcsec)")
    print(f"  ERA = {np.degrees(era):.9f}°")
    print(f"  EO = {eo/DAS2R:.6f} arcsec")
    print(f"  dpsi = {dpsi/DAS2R:.9f} arcsec")
    print(f"  deps = {deps/DAS2R:.9f} arcsec")
    print(f"  eps_A = {eps_a/DAS2R:.6f} arcsec")


# ===========================================================================
# Test Case 3: GPS Timestamp Test Cases
# ===========================================================================

def test_gps_timestamps(tr):
    """Test the full transformation pipeline using GPS timestamps.

    Converts GPS time → UTC → TT → transformation, and verifies
    against ERFA reference computed from the same JD values.
    """
    print("\n--- Test 9: GPS timestamp conversion and transformation ---")

    # ERFA test vectors (from t_erfa_c.c)
    # ---

    # Test cases: (GPS seconds, xp, yp, description)
    # GPS epoch: 1980-01-06T00:00:00 UTC (leap_seconds=19, DUT1≈0)
    test_cases = [
        # GPS second 0 = 1980-01-06 00:00:00 UTC
        (0.0, 0.0, 0.0, "GPS epoch 1980-01-06"),

        # 2006-01-01 00:00:00 UTC
        # GPS = 820108796 (GPS is 14s ahead of UTC: 820108796 - 14 = 820108782 UTC sec)
        # Actually GPS second for 2006-01-01 00:00 UTC:
        # From 1980-01-06 to 2006-01-01 = 9493 days = 820099200 sec
        # GPS = 820099200 (with leap_seconds=33, GPS-UTC=14)
        (820099200.0, 2.55060238e-7, 1.860359247e-6,
         "2006-01-01 (ERFA test date approx)"),

        # IAU 2006 paper date: 2006-01-15 21:24:37.5 UTC
        # GPS seconds = 821395491.5 (GPS = UTC + 14s)
        (821395491.5, 0.0, 0.0, "2006-01-15 (IAU2006 paper date)"),

        # J2000.0 epoch: 2000-01-01 12:00:00 TT = 11:58:55.816 UTC
        # GPS seconds = 630763200 + 14 = 630763214 (GPS-UTC=13 in 2000)
        (630763214.0, 0.0, 0.0, "J2000.0 epoch"),
    ]

    for gps_sec, xp, yp, desc in test_cases:
        print(f"\n  Case: {desc}")
        print(f"    GPS seconds = {gps_sec}")

        # Convert GPS → UTC → TT
        utc_jd = gps_to_utc(gps_sec)
        tt_jd1, tt_jd2 = utc_to_tt(utc_jd)
        # For UT1, use UTC (assuming DUT1=0 for simplicity)
        ut1_jd1, ut1_jd2 = DJM0, utc_jd - DJM0

        print(f"    UTC JD = {utc_jd:.10f}")
        print(f"    TT JD = {tt_jd1 + tt_jd2:.10f}")

        # Our transformation
        t2c, info = itrs_to_gcrs(tt_jd1, tt_jd2, ut1_jd1, ut1_jd2, xp, yp)

        # ERFA reference
        rc2t = erfa.c2t06a(tt_jd1, tt_jd2, ut1_jd1, ut1_jd2, xp, yp)

        # Compare
        diff = np.max(np.abs(t2c - rc2t.T))
        tr.check_true(f"GPS→ITRS→GCRS matches ERFA ({desc})",
                     diff < 1e-12, f"max diff={diff:.2e}")

        # Test a sample vector
        r_itrs = np.array([6378137.0, 0.0, 0.0])  # point on equator
        r_gcrs_ours = t2c @ r_itrs
        r_gcrs_erfa = rc2t.T @ r_itrs
        vec_diff = np.max(np.abs(r_gcrs_ours - r_gcrs_erfa))
        tr.check_true(f"Vector transform matches ({desc})",
                     vec_diff < 1e-6, f"diff={vec_diff:.2e} m")


# ===========================================================================
# Test Case 4: Common Pitfall Detection
# ===========================================================================

def test_pitfall_matrix_order(tr):
    """Pitfall: Wrong matrix multiplication order.

    The correct ITRS→GCRS order is Q·R·W, where:
      Q = rc2i.T  (CIRS→GCRS)
      R = Rz(-ERA) (TIRS→CIRS)
      W = rpom.T  (ITRS→TIRS)

    This is equivalent to (W_c2t · Rz(ERA) · Q_c2i)^T = C2T^T.

    Common mistakes:
    1. Forgetting to transpose: using C2T components (W_c2t·Rz(ERA)·Q_c2i)
       as the ITRS→GCRS matrix. This gives the GCRS→ITRS matrix instead.
       Error: ~2 (completely wrong direction)
    2. Wrong order of T2C components: W·R·Q instead of Q·R·W.
       Error: small (~1e-4) because matrices are nearly identity, but still wrong.
    3. Other permutations of component order (Q·W·R, R·Q·W, etc.)
       Error: small but non-zero.
    """
    print("\n--- Test 10: Pitfall — Wrong matrix multiplication order ---")

    # Use the ERFA test date
    tta, ttb = 2400000.5, 53736.0
    uta, utb = 2400000.5, 53736.0
    xp, yp = 2.55060238e-7, 1.860359247e-6

    # Get component matrices
    x, y, s = erfa.xys06a(tta, ttb)
    rc2i = erfa.c2ixys(x, y, s)      # GCRS→CIRS (Q_c2i)
    era = erfa.era00(uta, utb)
    sp = erfa.sp00(tta, ttb)
    rpom = erfa.pom00(xp, yp, sp)    # TIRS→ITRS (W_c2t)

    Q = rc2i.T    # CIRS→GCRS
    R = rz(-era)  # TIRS→CIRS
    W = rpom.T    # ITRS→TIRS

    # Reference: ERFA c2t06a
    rc2t = erfa.c2t06a(tta, ttb, uta, utb, xp, yp)
    t2c_ref = rc2t.T

    # --- Correct order: Q · R · W ---
    t2c_correct = Q @ R @ W
    diff_correct = np.max(np.abs(t2c_correct - t2c_ref))
    tr.check_true("Correct order Q·R·W", diff_correct < 1e-14,
                 f"diff={diff_correct:.2e}")

    # --- Pitfall 1: Forgetting transpose — using C2T as T2C ---
    # This is the most common and most damaging error.
    # C2T = rpom @ rz(era) @ rc2i  (GCRS→ITRS)
    # Using it directly as T2C gives the WRONG direction.
    t2c_wrong1 = rpom @ rz(era) @ rc2i  # C2T, not T2C!
    diff_wrong1 = np.max(np.abs(t2c_wrong1 - t2c_ref))
    tr.check_true("Pitfall: C2T used as T2C (no transpose)",
                 diff_wrong1 > 0.1,
                 f"diff={diff_wrong1:.2e} (should be > 0.1)")

    # --- Pitfall 2: Wrong order of T2C components: W·R·Q instead of Q·R·W ---
    # With nearly-identity matrices, this produces a small but non-zero error.
    # The error is proportional to the cross-products of off-diagonal terms.
    t2c_wrong2 = W @ R @ Q
    diff_wrong2 = np.max(np.abs(t2c_wrong2 - t2c_ref))
    tr.check_true("Pitfall: T2C components in wrong order W·R·Q (should be Q·R·W)",
                 diff_wrong2 > 1e-8,
                 f"diff={diff_wrong2:.2e}")

    # --- Pitfall 3: Q·W·R (wrong component order) ---
    t2c_wrong3 = Q @ W @ R
    diff_wrong3 = np.max(np.abs(t2c_wrong3 - t2c_ref))
    tr.check_true("Pitfall: order Q·W·R (wrong component order)",
                 diff_wrong3 > 1e-8,
                 f"diff={diff_wrong3:.2e}")

    # --- Pitfall 4: R·Q·W (wrong component order) ---
    t2c_wrong4 = R @ Q @ W
    diff_wrong4 = np.max(np.abs(t2c_wrong4 - t2c_ref))
    tr.check_true("Pitfall: order R·Q·W (wrong component order)",
                 diff_wrong4 > 1e-8,
                 f"diff={diff_wrong4:.2e}")

    # Demonstrate the error magnitude for a sample vector
    # The most damaging error (no transpose) gives ~14000 km error
    r_itrs = np.array([7000000.0, 0.0, 0.0])
    r_correct = t2c_correct @ r_itrs
    r_wrong_notranspose = t2c_wrong1 @ r_itrs
    error_meters = np.max(np.abs(r_wrong_notranspose - r_correct))
    print(f"  Vector error from missing transpose: {error_meters:.0f} m "
          f"({error_meters/1000:.0f} km)")
    print(f"  Note: wrong T2C component order errors are small ({diff_wrong2:.1e})")
    print(f"    because matrices are nearly identity, but still wrong.")


def test_pitfall_era_sign(tr):
    """Pitfall: Wrong sign of Earth Rotation Angle.

    In the ITRS→GCRS direction:
      R = R₃(-ERA)  (negative ERA, because we're going TIRS→CIRS, the reverse)

    A common mistake is to use R₃(+ERA) for the ITRS→GCRS direction.
    """
    print("\n--- Test 11: Pitfall — Wrong ERA sign in ITRS→GCRS ---")

    tta, ttb = 2400000.5, 53736.0
    uta, utb = 2400000.5, 53736.0
    xp, yp = 2.55060238e-7, 1.860359247e-6

    x, y, s = erfa.xys06a(tta, ttb)
    rc2i = erfa.c2ixys(x, y, s)
    era = erfa.era00(uta, utb)
    sp = erfa.sp00(tta, ttb)
    rpom = erfa.pom00(xp, yp, sp)

    Q = rc2i.T
    W = rpom.T

    # Reference
    rc2t = erfa.c2t06a(tta, ttb, uta, utb, xp, yp)
    t2c_ref = rc2t.T

    # Correct: R = Rz(-ERA)
    R_correct = rz(-era)
    t2c_correct = Q @ R_correct @ W
    diff_correct = np.max(np.abs(t2c_correct - t2c_ref))
    tr.check_true("Correct sign: Rz(-ERA)", diff_correct < 1e-14,
                 f"diff={diff_correct:.2e}")

    # Wrong: R = Rz(+ERA)
    R_wrong = rz(era)
    t2c_wrong = Q @ R_wrong @ W
    diff_wrong = np.max(np.abs(t2c_wrong - t2c_ref))
    tr.check_true("Pitfall: Rz(+ERA) in ITRS→GCRS direction",
                 diff_wrong > 0.1,
                 f"diff={diff_wrong:.2e} (should be > 0.1)")

    # The error should be approximately 2*ERA worth of rotation
    r_itrs = np.array([7000000.0, 0.0, 0.0])
    r_correct = t2c_correct @ r_itrs
    r_wrong_vec = t2c_wrong @ r_itrs
    error_meters = np.max(np.abs(r_wrong_vec - r_correct))
    print(f"  Vector error from wrong ERA sign: {error_meters:.0f} m "
          f"({error_meters/1000:.0f} km)")
    print(f"  (Expected ~ Earth radius * 2*sin(ERA) ≈ "
          f"{6378137 * 2 * abs(np.sin(era)):.0f} m)")


def test_pitfall_cio_locator_s(tr):
    """Pitfall: Confusing CIO locator s with Equation of the Origins (EO).

    The CIO locator s positions the Celestial Intermediate Origin on the
    equator of the CIP. It is NOT the same as the Equation of the Origins.

    - s: CIO locator (used in CIO-based transformation, appears in Q matrix)
    - EO: Equation of the Origins = ERA - GST (difference between ERA and
      Greenwich Sidereal Time, used in equinox-based transformation)
    - EO ≈ s + XY/2 (but they are NOT identical)

    Using EO instead of s (or vice versa) in the CIO-based transformation
    introduces errors.
    """
    print("\n--- Test 12: Pitfall — CIO locator s vs Equation of Origins ---")

    # IAU 2006 paper test date
    tt1, tt2 = 2400000.5, 53750.892855138888889
    ut1, ut2 = 2400000.5, 53750.892104561342593

    # Get s and EO
    x, y, s = erfa.xys06a(tt1, tt2)
    eo = erfa.eo06a(tt1, tt2)

    print(f"  CIO locator s = {s:.15e} rad ({s/DAS2R:.9f} arcsec)")
    print(f"  Equation of Origins EO = {eo:.15e} rad ({eo/DAS2R:.6f} arcsec)")
    print(f"  Difference (EO - s) = {eo - s:.6e} rad ({(eo-s)/DAS2R:.6f} arcsec)")

    # They are very different in magnitude!
    tr.check_true("s ≠ EO (different magnitudes)",
                 abs(eo - s) > abs(s) * 100,
                 f"|EO-s|={abs(eo-s):.2e}, |s|={abs(s):.2e}")

    # Test: using EO instead of s in the Q matrix
    # Correct Q uses s
    rc2i_correct = erfa.c2ixys(x, y, s)

    # Wrong: using EO instead of s
    rc2i_wrong = erfa.c2ixys(x, y, eo)

    diff = np.max(np.abs(rc2i_correct - rc2i_wrong))
    tr.check_true("Using EO instead of s gives wrong Q matrix",
                 diff > 1e-6,
                 f"max diff={diff:.2e} (should be > 1e-6)")

    # The error propagates to the full transformation
    era = erfa.era00(ut1, ut2)
    sp = erfa.sp00(tt1, tt2)
    xp, yp = 0.0, 0.0
    rpom = erfa.pom00(xp, yp, sp)

    # Correct
    t2c_correct = rc2i_correct.T @ rz(-era) @ rpom.T

    # Wrong (using EO for s)
    t2c_wrong = rc2i_wrong.T @ rz(-era) @ rpom.T

    diff_full = np.max(np.abs(t2c_correct - t2c_wrong))
    tr.check_true("Full transform error from s→EO substitution",
                 diff_full > 1e-6,
                 f"max diff={diff_full:.2e}")

    # Reference check
    rc2t = erfa.c2t06a(tt1, tt2, ut1, ut2, xp, yp)
    tr.check_matrix("Correct s-based transform vs ERFA",
                    t2c_correct, rc2t.T, tol=1e-14)

    # The relationship between EO and s:
    # s is the CIO locator used in the CIO-based Q matrix.
    # EO is the Equation of the Origins, used in equinox/sidereal-time formulations.
    # EO = ERA - GAST (exact). They are NOT interchangeable.
    # EO includes the full precession-nutation, while s is a small angle (~µarcsec).
    # Do NOT substitute one for the other.
    print(f"  |EO| = {abs(eo)/DAS2R:.6f} arcsec, |s| = {abs(s)/DAS2R:.9f} arcsec")
    print(f"  |s + XY/2| = {abs(s + x*y/2)/DAS2R:.9f} arcsec (NOT equal to EO)")
    print(f"  Note: s is for CIO-based Q matrix; EO is for equinox/GST formulations")


def test_pitfall_polar_motion_units(tr):
    """Pitfall: Polar motion in arcseconds vs radians.

    IERS bulletins typically provide xp, yp in arcseconds.
    ERFA/SOFA expects radians. Forgetting to convert introduces
    errors of ~5 orders of magnitude.
    """
    print("\n--- Test 13: Pitfall — Polar motion units (arcsec vs rad) ---")

    tta, ttb = 2400000.5, 53736.0
    uta, utb = 2400000.5, 53736.0

    # ERFA test values (in radians)
    xp_rad = 2.55060238e-7    # radians
    yp_rad = 1.860359247e-6   # radians

    # Same values in arcseconds (what IERS bulletins provide)
    xp_arcsec = xp_rad / DAS2R   # ≈ 0.0526 arcsec
    yp_arcsec = yp_rad / DAS2R   # ≈ 0.384 arcsec

    print(f"  xp = {xp_rad:.6e} rad = {xp_arcsec:.6f} arcsec")
    print(f"  yp = {yp_rad:.6e} rad = {yp_arcsec:.6f} arcsec")

    # Correct: use radians
    t2c_correct, _ = itrs_to_gcrs(tta, ttb, uta, utb, xp_rad, yp_rad)
    rc2t = erfa.c2t06a(tta, ttb, uta, utb, xp_rad, yp_rad)
    tr.check_matrix("Correct: xp,yp in radians",
                    t2c_correct, rc2t.T, tol=1e-14)

    # Wrong: use arcseconds as if they were radians
    t2c_wrong, _ = itrs_to_gcrs(tta, ttb, uta, utb, xp_arcsec, yp_arcsec)
    diff = np.max(np.abs(t2c_correct - t2c_wrong))
    tr.check_true("Pitfall: passing arcseconds as radians",
                 diff > 0.1,
                 f"diff={diff:.2e} (should be > 0.1)")

    # The error is enormous
    r_itrs = np.array([6378137.0, 0.0, 0.0])
    r_correct = t2c_correct @ r_itrs
    r_wrong = t2c_wrong @ r_itrs
    error_meters = np.max(np.abs(r_correct - r_wrong))
    print(f"  Position error from unit confusion: {error_meters:.0f} m "
          f"({error_meters/1000:.0f} km)")


def test_pitfall_time_scales(tr):
    """Pitfall: Using GPS or UTC directly as TT or UT1.

    GPS time is NOT UTC (differs by leap seconds).
    UTC is NOT TT (differs by leap seconds + 32.184s).
    TT is NOT UT1 (differs by DUT1, typically < 1s).

    Using GPS as TT introduces an error of ~19-37 seconds,
    which translates to ~10,000+ meters in position.
    """
    print("\n--- Test 14: Pitfall — Time scale confusion ---")

    # IAU 2006 paper test date
    tt1_correct, tt2_correct = 2400000.5, 53750.892855138888889
    ut1_correct, ut2_correct = 2400000.5, 53750.892104561342593

    xp, yp = 0.0, 0.0

    # Correct transformation
    t2c_correct, _ = itrs_to_gcrs(tt1_correct, tt2_correct,
                                  ut1_correct, ut2_correct, xp, yp)

    # --- Pitfall 1: Using UTC as TT (error = 32.184 + leap_seconds ≈ 65s) ---
    # This affects the precession-nutation (Q matrix) and TIO locator (s').
    # The error is small for Q because precession-nutation changes slowly,
    # but the error compounds if TT is also used for UT1.
    utc_jd = ut1_correct + ut2_correct  # This is UTC JD
    tt_wrong_jd1 = DJM0
    tt_wrong_jd2 = utc_jd - DJM0  # Using UTC as TT!

    t2c_wrong1, _ = itrs_to_gcrs(tt_wrong_jd1, tt_wrong_jd2,
                                 ut1_correct, ut2_correct, xp, yp)
    diff1 = np.max(np.abs(t2c_correct - t2c_wrong1))
    # The error is small (~1e-10) because precession-nutation changes slowly
    # over 65 seconds. But it's still wrong.
    tr.check_true("Pitfall: using UTC as TT (small but non-zero error)",
                 diff1 > 1e-12,
                 f"diff={diff1:.2e}")

    # --- Pitfall 2: Using TT as UT1 (ignoring DUT1) ---
    # DUT1 ≈ 0.3341s for this date. This directly affects ERA.
    # Earth rotates ~15 arcsec/s, so 0.3341s → ~5 arcsec error → ~150m on ground
    t2c_wrong2, _ = itrs_to_gcrs(tt1_correct, tt2_correct,
                                 tt1_correct, tt2_correct, xp, yp)
    diff2 = np.max(np.abs(t2c_correct - t2c_wrong2))
    tr.check_true("Pitfall: using TT as UT1 (ignoring DUT1)",
                 diff2 > 1e-8,
                 f"diff={diff2:.2e}")

    # --- Pitfall 3: Using GPS as UTC (error = leap_seconds - 19) ---
    # GPS is ahead of UTC by (leap_seconds - 19) seconds.
    # For the IAU 2006 date (2006), leap_seconds = 33, so GPS-UTC = 14s.
    # Using GPS JD directly as UTC JD shifts everything by 14s.
    gps_seconds = 821395491.5  # GPS for IAU 2006 date
    gps_jd = GPS_JD0 + gps_seconds / 86400.0
    # Wrong: using GPS JD directly as UTC JD, then converting to TT
    tt_wrong3_jd1 = DJM0
    tt_wrong3_jd2 = gps_jd - DJM0 + (TT_MINUS_TAI + 33)/86400.0
    t2c_wrong3, _ = itrs_to_gcrs(tt_wrong3_jd1, tt_wrong3_jd2,
                                 DJM0, gps_jd - DJM0, xp, yp)
    diff3 = np.max(np.abs(t2c_correct - t2c_wrong3))
    tr.check_true("Pitfall: using GPS as UTC (14s error)",
                 diff3 > 1e-8,
                 f"diff={diff3:.2e}")

    # Error magnitude for a surface point
    r_itrs = np.array([6378137.0, 0.0, 0.0])
    r_correct = t2c_correct @ r_itrs
    r_wrong1 = t2c_wrong1 @ r_itrs
    r_wrong2 = t2c_wrong2 @ r_itrs

    err1 = np.max(np.abs(r_correct - r_wrong1))
    err2 = np.max(np.abs(r_correct - r_wrong2))
    r_wrong3 = t2c_wrong3 @ r_itrs
    err3 = np.max(np.abs(r_correct - r_wrong3))
    print(f"  Error from UTC→TT confusion: {err1:.1f} m (small: precession changes slowly)")
    print(f"  Error from ignoring DUT1: {err2:.0f} m ({err2/1000:.0f} km)")
    print(f"  Error from GPS→UTC confusion: {err3:.0f} m ({err3/1000:.0f} km)")
    print(f"  (UTC-TT error affects Q slowly; DUT1 error affects ERA directly)")


def test_pitfall_gast_vs_era(tr):
    """Pitfall: Confusing ERA (Earth Rotation Angle) with GAST/GMST.

    ERA is used in the CIO-based transformation.
    GAST (Greenwich Apparent Sidereal Time) is used in the equinox-based
    transformation.

    ERA = GMST + EO (approximately)
    ERA ≠ GAST ≠ GMST

    Using GAST where ERA is required introduces arcsecond-level errors.
    """
    print("\n--- Test 15: Pitfall — ERA vs GAST/GMST confusion ---")

    tt1, tt2 = 2400000.5, 53750.892855138888889
    ut1, ut2 = 2400000.5, 53750.892104561342593

    era = erfa.era00(ut1, ut2)
    gmst = erfa.gmst06(ut1, ut2, tt1, tt2)
    rbpn = erfa.pnm06a(tt1, tt2)
    # NOTE: gst06 argument order is (ut1a, ut1b, tta, ttb, rbpn) — UT1 first!
    gast = erfa.gst06(ut1, ut2, tt1, tt2, rbpn)
    eo = erfa.eo06a(tt1, tt2)

    print(f"  ERA  = {np.degrees(era):.9f}°")
    print(f"  GMST = {np.degrees(gmst):.9f}°")
    print(f"  GAST = {np.degrees(gast):.9f}°")
    print(f"  EO   = {np.degrees(eo):.6f}° ({np.degrees(eo)*3600:.6f} arcsec)")
    print(f"  ERA - GAST = {np.degrees(era - gast):.6f}° ({np.degrees(era-gast)*3600:.6f} arcsec)")
    print(f"  GAST - GMST = {(gast-gmst)/DAS2R:.6f} arcsec (equation of equinoxes)")

    # ERA ≠ GAST
    tr.check_true("ERA ≠ GAST",
                 abs(era - gast) > 1e-6,
                 f"|ERA-GAST|={abs(era-gast):.2e} rad")

    # ERA ≠ GMST
    tr.check_true("ERA ≠ GMST",
                 abs(era - gmst) > 1e-6,
                 f"|ERA-GMST|={abs(era-gmst):.2e} rad")

    # The EXACT relationship: EO = ERA - GAST
    # This is exact to machine precision when gst06 is called with
    # correct argument order (ut1, tt, rbpn).
    tr.check_true("EO = ERA - GAST (exact relationship)",
                 abs((era - gast) - eo) < 1e-14,
                 f"diff={abs((era-gast)-eo):.2e} rad")


def test_pitfall_transpose_confusion(tr):
    """Pitfall: Confusing the direction of the transformation.

    SOFA's c2t06a returns the GCRS→ITRS matrix (celestial-to-terrestrial).
    To get ITRS→GCRS, you must TRANSPOSE it.

    Many implementations forget the transpose, leading to a matrix
    that rotates in the wrong direction.
    """
    print("\n--- Test 16: Pitfall — Forgetting transpose (C2T vs T2C) ---")

    tta, ttb = 2400000.5, 53736.0
    uta, utb = 2400000.5, 53736.0
    xp, yp = 2.55060238e-7, 1.860359247e-6

    rc2t = erfa.c2t06a(tta, ttb, uta, utb, xp, yp)  # GCRS→ITRS

    # Correct: ITRS→GCRS = C2T^T
    t2c_correct = rc2t.T

    # Wrong: Using C2T directly as ITRS→GCRS
    t2c_wrong = rc2t  # No transpose!

    diff = np.max(np.abs(t2c_correct - t2c_wrong))
    tr.check_true("C2T ≠ T2C (transpose needed)",
                 diff > 0.1,
                 f"diff={diff:.2e}")

    # The wrong matrix is the transpose, so it rotates in the opposite direction
    r_itrs = np.array([7000000.0, 0.0, 0.0])
    r_correct = t2c_correct @ r_itrs
    r_wrong = t2c_wrong @ r_itrs
    error = np.max(np.abs(r_correct - r_wrong))
    print(f"  Position error from missing transpose: {error:.0f} m "
          f"({error/1000:.0f} km)")

    # The correct result should match our implementation
    t2c_ours, _ = itrs_to_gcrs(tta, ttb, uta, utb, xp, yp)
    tr.check_matrix("Our implementation = c2t06a().T",
                    t2c_ours, rc2t.T, tol=1e-14)


# ===========================================================================
# Main test runner
# ===========================================================================

def run_all_tests():
    """Run the complete test suite."""
    tr = TestResults()

    print("=" * 70)
    print("ITRS-to-GCRS Frame Transformation Verification Suite")
    print("IERS Conventions 2010 / IAU 2006/2000A")
    print("=" * 70)

    # ERFA official test vectors
    test_erfa_era00(tr)
    test_erfa_sp00(tr)
    test_erfa_xys06a(tr)
    test_erfa_s06a(tr)
    test_erfa_pom00(tr)
    test_erfa_c2t06a(tr)
    test_erfa_c2tcio(tr)

    # IAU 2006 reference values
    test_iau2006_reference_values(tr)

    # GPS timestamp tests
    test_gps_timestamps(tr)

    # Pitfall detection tests
    test_pitfall_matrix_order(tr)
    test_pitfall_era_sign(tr)
    test_pitfall_cio_locator_s(tr)
    test_pitfall_polar_motion_units(tr)
    test_pitfall_time_scales(tr)
    test_pitfall_gast_vs_era(tr)
    test_pitfall_transpose_confusion(tr)

    # Print details
    print("\n" + "-" * 70)
    print("DETAILED RESULTS:")
    print("-" * 70)
    for d in tr.details:
        print(d)

    print(tr.summary())

    return tr


if __name__ == "__main__":
    tr = run_all_tests()

# %%
