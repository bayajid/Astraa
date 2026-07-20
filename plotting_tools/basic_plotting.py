import numpy as np
import matplotlib.pyplot as plt
import os, traceback, inspect
import datetime as dt
import sys
import pickle
import git
import platform
import subprocess
def autosave(fig,
            save_folder=r'outputs/plots',
            adjust_name = 1,
            subfolder = 'untagged',
            timetag = 0,
        ):
        full_folder = f'{os.getcwd()}//{save_folder}//{subfolder}'
        name = fig._suptitle.get_text()
        if adjust_name:
                name = name.replace(r'\n','')
                name = name.replace(' ', '_')
                name = name.replace('-', '_')                
        if timetag:     
                # add DDD_HH_MM time-tag
                yrday_hmin = dt.datetime.now().strftime('%j_%H_%M')
                name = f'{name}{yrday_hmin}'
        fig_name = f'{full_folder}//{name}.png'        
        try:
                fig.savefig(fig_name, bbox_inches='tight')
                print(f'---Saved {name} in {save_folder}---')
        except:
                # Try to create subfolder
                print('---Trying to create subfolder---')
                os.mkdir(full_folder)
                print(f'---Successfully created {save_folder}//{subfolder}---')
                fig.savefig(fig_name, bbox_inches='tight')
                print(f'---Saved {name} in {subfolder}---')
        
def savefig(fig,
            name,
            save_folder=r'outputs/plots',
            adjust_name = 1,
            subfolder = None,
            timetag = 0,
            add_script_path = 1,
            x_coord_tag = 1.1,
            y_coord_tag = -2.5,
            open_folder = 1,
            save = 1,
            tag_option = 0,
            save_as_matfig = 0,
            
            ):
        # Saves figure. If subfolder does not exist, attempts to cerate it first.
        # can optinally add time tag with timetag = 1
        if type(subfolder) != type(None):
                full_folder = f'{os.getcwd()}//{save_folder}//{subfolder}'
        else:
                full_folder = f'{os.getcwd()}//{save_folder}//'

        if adjust_name:
                name = name.replace(' ', '_')
                name = name.replace('-', '_')                
        if timetag:     
                # add DDD_HH_MM time-tag
                yrday_hmin = dt.datetime.now().strftime('%j_%H_%M')
                name = f'{name}{yrday_hmin}'
        if add_script_path: # Add path/git commit hash and timestamp
                # Find path of script that called the savefig to print on figure
                
                script_name = 'basic_plotting.py'
                stack_1 = inspect.stack()
                stacks_w_apy = []
                print(stack_1)
                for ii, path_obj in enumerate(stack_1):
                        if 'astropynaric'in path_obj.filename:
                                stacks_w_apy.append(path_obj.filename)
                                # ii_parent = ii + 1
                                # break
                print(stacks_w_apy)
                try:
                        path_stamp = [p for p in stacks_w_apy if script_name not in p][0]
                        path_stamp = path_stamp[path_stamp.index('astropynaric'):]
                except:
                        path_stamp = ''
                # get git hash
                repo = git.Repo(search_parent_directories=True)
                sha = repo.head.object.hexsha
                date_stamp = dt.datetime.now().strftime('%Y_%m_%d')
                ## Add 
                ax = fig.axes
                text = f'{path_stamp} created on {date_stamp}\nGL SHA: {sha}'
                # ax[0].text(x = 0.25, y = 1.2, s = text, bbox=dict(facecolor='white'))                
                
                if tag_option == 0:
                        xlabel = ax[0].get_xlabel()
                        ax[0].set_xlabel(f'{xlabel}\n{text}')
                elif tag_option == 1:
                        fig.suptitle(text, fontsize = 9)
                elif tag_option == 2:
                        ax[0].text(x = x_coord_tag, y = y_coord_tag, s = text, bbox=dict(facecolor='white', ), fontsize = 7, rotation = 'horizontal')
                        
                
                
                
                
        fig_name = fr'{full_folder}/{name}.png'        
        if save:
                try:
                        fig.set_tight_layout('tight')
                        fig.savefig(fig_name, bbox_inches='tight')
                        print(f'---Saved {name} in {save_folder}---')
                        try:
                                if open_folder:
                                        if platform.system() == 'Windows':
                                                os.startfile(os.path.realpath(full_folder))
                                        else:
                                                subprocess.run(["xdg-open", full_folder])
                        except:
                                pass
                        
                except:
                        # Try to create subfolder
                        print('---Trying to create subfolder---')
                        os.mkdir(full_folder)
                        print(f'---Successfully created {save_folder}//{subfolder}---')
                        fig.savefig(fig_name, bbox_inches='tight')
                        print(f'---Saved {name} in {subfolder}---')
                        try:
                                if open_folder:
                                        if platform.system() == 'Windows':
                                                os.startfile(os.path.realpath(full_folder))
                                        else:
                                                subprocess.run(["xdg-open", full_folder])
                        except:
                                pass
                if save_as_matfig:
                        with open(f'{fig_name[:-4]}.fig', 'wb') as f:
                                pickle.dump(plt.gcf(), f)


if __name__ == '__main__':
        
        f, ax = plt.subplots()
        txt = 'astropynaric\\analyses\\on_orbit_calibration\\MOQ_resolution_sims\\analyze_moon_illum.py'
        txt2 = 'stropynaric\\analyses\\on_orbit_calibration\\MOQ_re'
        txt3 = dt.datetime.now().strftime('%Y-%m-%d')
        
        text = f'{txt}\n{txt2} created on {txt3}'
        ax.text(x = 1.1 ,y = -0.1, s = text, bbox=dict(facecolor='white', ), fontsize = 7, rotation = 'vertical')
        ax.set_xlabel('time since start [s]')
        ax.set_ylabel('time since start [s]')