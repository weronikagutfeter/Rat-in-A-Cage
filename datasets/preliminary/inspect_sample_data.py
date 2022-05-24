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

def plot_ts(srckey,valkey,timestamps, track_coords,xlabel='timestamp'):
    fig, axs = plt.subplots(2,figsize=(16, 8))
    name = srckey + ' ' + valkey
    fig.suptitle(name)

    plt.xlabel(xlabel)
    axs[0].set_ylabel('x' + valkey.capitalize())
    axs[1].set_ylabel('y' + valkey.capitalize())

    for c, (ts, coords) in enumerate(zip(timestamps,track_coords)):

        x = coords[:,0]
        y = coords[:,1]

        axs[0].scatter(ts,x,s=3,c=[colors[int(c%9)]])
        axs[1].scatter(ts, y, s=3, c=[colors[int(c % 9)]])

    return plt

def plot_dt(srckey,valkey,timestamps, track_coords,xlabel='timestamp'):
    plt.figure(figsize=(16, 6))
    name = srckey + ' ' + valkey
    plt.title(name)

    plt.xlabel(xlabel)
    plt.ylabel(valkey.capitalize())

    plt.scatter(timestamps,track_coords,s=3,c=[colors[0]])

    return plt

def plot_histos(name, data_dict):
    plt.figure(figsize=(16, 6))
    plt.title(name)

    dt = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

    bins = np.concatenate((dt,[np.inf]))#np.logspace(-2, 3, 10, endpoint=True)
    inds = np.arange(len(bins))
    n_keys = len(data_dict)
    width = 1.0/(n_keys+1)


    for c, (key,values) in enumerate(data_dict.items()):
        n = len(values)
        hist, bin_edges = np.histogram(values, bins)
        hist = hist/n

        plt.bar(inds[:-1]+c*width,hist,width,color=[colors[int(c%9)]], label=key)


    plt.xticks([ind-0.5*width for ind in inds[:-1]],bins[:-1])
    plt.legend()
    return plt

def dist(a,b):
    return np.sqrt(np.sum((a-b)**2))

def transform2deriv(track_xy,track_timestamps,make_flat=True):
    if(make_flat):
        flat_xy = [xy for track in track_xy for xy in track]
        flat_time = [t for track in track_timestamps for t in track]
    else:
        flat_xy = track_xy
        flat_time = track_timestamps
    velo = []
    dt = []
    for k in range(1,len(flat_xy)):
        if(flat_time[k]-flat_time[k-1]>0):
            velo.append(dist(flat_xy[k],flat_xy[k-1]) / (flat_time[k] - flat_time[k - 1]))
            dt.append(flat_time[k])

    return velo, dt

def main(jsonfile, outputdir=None):

    timestamp_key = "tsFromEvent"

    all_velocities = {key:[] for key in access_helper.keys()}
    all_accelerations = {key:[] for key in access_helper.keys()}

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

            if(True):#if(flag<5 and flag>3):

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
                        timestamps.append(entry[timestamp_key])


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

                velocity, dtv = transform2deriv(track_screen, track_timestamps, make_flat=True)
                accelaration, dta = transform2deriv(velocity, dtv, make_flat=False)

                all_velocities[key].extend(velocity)
                all_accelerations[key].extend(accelaration)

                # if(np.all(np.array(list(access_helper.values()))>1)):
                #     break

                if (False):#(flag < 5 and flag > 3):
                    # plot_XY('offsets '+key,track_offsets)
                    # plot_XY('client',cClient)
                    # plot_XY('screen '+key, track_screen)
                    # if(outputdir is not None):
                    #     plt.savefig(os.path.join(outputdir,'screen-'+key+'.png'))
                    # # plot_XY('page '+key, track_page)
                    # plot_XY('diffs '+key, track_diffs)
                    # if (outputdir is not None):
                    #     plt.savefig(os.path.join(outputdir, 'diffs-' + key + '.png'))

                    # plot_ts('diffs'+key,track_timestamps,track_diffs)
                    plot_ts(key,'screen', track_timestamps, track_screen,xlabel=timestamp_key)
                    if (outputdir is not None):
                        plt.savefig(os.path.join(outputdir, 'screen-ts-' + key + '.png'))

                    plot_ts(key,'Movement', track_timestamps, track_diffs,xlabel=timestamp_key)


                    plot_dt(key,'Velocity', dtv, velocity,xlabel=timestamp_key)
                    if (outputdir is not None):
                        plt.savefig(os.path.join(outputdir, 'velocity-' + key + '.png'))

                    plot_dt(key, 'Acceleration', dta, accelaration, xlabel=timestamp_key)
                    if (outputdir is not None):
                        plt.savefig(os.path.join(outputdir, 'acceleration-' + key + '.png'))

    for key,value in access_helper.items():
        print("{}: {} samples".format(key,value))

    plot_histos('Mouse velocity',all_velocities)
    if (outputdir is not None):
        plt.savefig(os.path.join(outputdir, 'hist-velocity.png'))

    plot_histos('Mouse acceleration', all_accelerations)
    if (outputdir is not None):
        plt.savefig(os.path.join(outputdir, 'hist-acceleration.png'))

    plt.show()






if __name__ == '__main__':

    jsonfile = '../../sample_data/cumulative_micro_times_devT2_all.json'
    outputdir = '/home/weronika/Output/Ratinacage'
    main(jsonfile,outputdir)