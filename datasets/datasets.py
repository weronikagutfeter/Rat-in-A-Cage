import json
import numpy as np
import os
import pickle

category_helper = {'genuine' : 0,
                 'anydesk': 1,
                 'teamviewer' : 2,
                 'chromeremote': 3}


def load_json(jsonfile,coordskey='Screen',timekey='ts'):
    with open(jsonfile,'r') as jf:
        mouse_json = json.load(jf)

        session_data = []

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


            track_data = []
            track_timestamps = []

            for track in mouse_moves:
                print("\t - track len {}".format(len(track)))
                for entry in track:
                    track_data.append((entry['x'+coordskey],entry['y'+coordskey]))
                    track_timestamps.append(entry[timekey])


            track_data = np.array(track_data)
            track_timestamps = np.array(track_timestamps)

            session_data.append({
                'category':key,
                'catid': category_helper[key],
                'data':track_data,
                'time':track_timestamps

            })


    return session_data


def make_folds(jsonfile, outputdir,coordskey='Screen',timekey='ts',tracklen=30):

    sessions = load_json(jsonfile,coordskey,timekey)

    idxs= np.arange(len(sessions))
    labels = [s['category']=='genuine' for s in sessions]
    groups = [s['catid'] for s in sessions]

    from sklearn.model_selection import StratifiedShuffleSplit


    for fold, (train_idxs, test_idxs) in enumerate(StratifiedShuffleSplit(n_splits=3,test_size=0.2).split(idxs,groups)):
        train_sessions = [sessions[sid] for sid in train_idxs]
        for ts in train_sessions:
            ts['data'] = ts['data'][:tracklen]
            ts['time'] = ts['time'][:tracklen]
        test_sessions = [sessions[sid] for sid in test_idxs]
        for ts in test_sessions:
            ts['data'] = ts['data'][:tracklen]
            ts['time'] = ts['time'][:tracklen]

        picklename = os.path.join(outputdir,"{}_{}_fold{}{}.pkl".format(coordskey,timekey,fold,'_redto'+str(tracklen) if tracklen > 0 else ''))

        with open(picklename, 'wb') as f:
            pickle.dump((train_sessions,test_sessions),f,-1)





if __name__ == '__main__':
    jsonfile = '../sample_data/cumulative_micro_times_devT2_all.json'
    outputdir = '../sample_data/folds'
    make_folds(jsonfile,outputdir)

