import json
import matplotlib.pyplot as plt
import numpy as np
import os

from matplotlib import cm
cmap = cm.get_cmap('Set1')
colors = [np.array(cmap(x/9.0)) for x in range(9)]

access_helper = {'genuine' : 0,
                 'anydesk': 0,
                 'teamviewer' : 0,
                 'chromeremote': 0}

def plot_XY(name,track_coords):
    plt.figure(figsize=(8, 6))
    plt.title(name)
    plt.grid()

    for c, coords in enumerate(track_coords):

        x = coords[:,0]
        y = coords[:,1]

        plt.plot(x,y,color=colors[int(c%9)])

    return plt

def plot_ts(name,timestamps, track_coords):
    fig, axs = plt.subplots(2,figsize=(16, 8))
    fig.suptitle(name)

    for c, (ts, coords) in enumerate(zip(timestamps,track_coords)):

        x = coords[:,0]
        y = coords[:,1]

        axs[0].scatter(ts,x,s=3,c=[colors[int(c%9)]])
        axs[1].scatter(ts, y, s=3, c=[colors[int(c % 9)]])

    return plt

def main(jsonfile, outputdir=None):

    with open(jsonfile,'r') as jf:
        mouse_json = json.load(jf)

        for mouse_session in mouse_json:
            session_name = mouse_session['sid']

            mouse_moves = mouse_session['mouse_move']
            print('{} tracks in mouse move {}'.format(len(mouse_moves),session_name))

            if(mouse_session['isRemote']):
                if(mouse_session['typeRemoteAccess']['isAnyDesk']):
                    key ='anydesk'
                elif(mouse_session['typeRemoteAccess']['isTeamViewer']):
                    key = 'teamviewer'
                else:
                    key = 'chromeremote'
            else:
                key = 'genuine'

            flag = access_helper[key]
            access_helper[key] += 1

            if(flag<5 and flag>3):

                track_offsets = []
                track_page = []
                track_screen = []
                track_diffs = []

                track_timestamps = []

                for track in mouse_moves:
                    print("\t - track len {}".format(len(track)))
                    offsets = []
                    cClient = []
                    cScreen = []
                    cPage = []
                    diffs = []
                    timestamps = []
                    for entry in track:
                        offsets.append((entry['xOffset'],entry['yOffset']))
                        cClient.append((entry['xClient'], entry['yClient']))
                        cScreen.append((entry['xScreen'], entry['yScreen']))
                        cPage.append((entry['xPage'], entry['yPage']))
                        diffs.append((entry['xMovement'], entry['yMovement']))
                        timestamps.append(entry['tsFromEvent'])


                    offsets = np.array(offsets)
                    cClient = np.array(cClient)
                    cScreen = np.array(cScreen)
                    cPage = np.array(cPage)
                    diffs = np.array(diffs)
                    timestamps = np.array(timestamps)

                    track_offsets.append(offsets)
                    track_screen.append(cScreen)
                    track_page.append(cPage)
                    track_diffs.append(diffs)

                    track_timestamps.append(timestamps)


                # plot_XY('offsets '+key,track_offsets)
                # plot_XY('client',cClient)
                plot_XY('screen '+key, track_screen)
                if(outputdir is not None):
                    plt.savefig(os.path.join(outputdir,'screen-'+key+'.png'))
                # plot_XY('page '+key, track_page)
                plot_XY('diffs '+key, track_diffs)
                if (outputdir is not None):
                    plt.savefig(os.path.join(outputdir, 'diffs-' + key + '.png'))

                # plot_ts('diffs'+key,track_timestamps,track_diffs)
                plot_ts('screen ' + key, track_timestamps, track_screen)
                if (outputdir is not None):
                    plt.savefig(os.path.join(outputdir, 'screen-ts-' + key + '.png'))

    for key,value in access_helper.items():
        print("{}: {} samples".format(key,value))

    plt.show()






if __name__ == '__main__':

    jsonfile = '../../sample_data/cumulative_micro_times_devT2_all.json'
    outputdir = '/home/weronika/Output/Ratinacage'
    main(jsonfile,outputdir)