# File to defien the LEO-LEO and LEO-MEO 
# hosts/targets labelled according to TUDAT constellation simulation parameters
case_titles =['Coplanar LEO P', 
              'Coplanar LEO I', 
              'Coplanar MEO',
                'Crossplane LEO I - LEO I', 
                'Crossplane LEO P - LEO P', #5
                'Crossplane LEO I - LEO I opposite', 
                'Crossplane LEO P - LEO P opposite', 
                'Crossplane LEO I - LEO P opposite', #8
                'LEO I - MEO', 
                'LEO P - MEO'
                ]
case_titles_brief =['Copl LEO P', 
              'Copl LEO I', 
              'Copl MEO',
                'Crosspl LEO I', 
                'Crosspl LEO P', #5
                'Crosspl LEO I', 
                'Crosspl LEO P', 
                'LEO I - P ', #8
                'LEO I - MEO', 
                'LEO P - MEO'
                ]
hosts = ['sat_leo_polar_0_4', 
         'sat_leo_incl_0_4', 
         'sat_meo_0_0', 
        'sat_leo_incl_0_4',
        'sat_leo_polar_0_4', #5
        'sat_leo_incl_0_4',
        'sat_leo_polar_0_4', 
        'sat_leo_incl_0_4',
        'sat_leo_incl_2_4',
        'sat_leo_polar_4_4', 
        ]
targets = ['sat_leo_polar_0_5', 'sat_leo_incl_0_5', 'sat_meo_0_1',
        'sat_leo_incl_1_4',
        'sat_leo_polar_1_4', #5
        'sat_leo_incl_7_7',
        'sat_leo_polar_7_0', 
        'sat_leo_polar_7_7', #8
        'sat_meo_0_0',
        'sat_meo_0_1',
        ]