from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
import argparse


class AttachmentPersonTracking(object):
    """ Person tracking Results of Infants and Mothers based on the Kinect Data

    The data is expected to be saved as a csv file with the first line as the header.
    The expected column names are
        1. "ID" for the session id;
        2. "RE" for which reunion the data is related to, either 1 or 2;
        3. "timestamp" for the time when the data point is collected;
        4. "{b, m}Pos3D_{X, Y, Z}" for the 3D tracking results of infants and mothers;
        5. "bHeightFromFloor" for the normalized height of baby from the floor;
        6. "{b, m}InterpolationType" for the indicator of missing data.
            1 for data missing in the beginning, which are filled by first seen data
            2 for normally collected data
            3 for data missing in the end, which are filled by last seen data
            999 for completely missing data
    """

    DEFAULT_CONTACT_THRESHOLD = 800
    DEFAULT_CARRY_THRESHOLD = 900
    DEFAULT_INIT_APP_TIME = 125

    def __init__(self, data_path: Path, output_path: Path):
        self.data = pd.read_csv(data_path)
        self._compute_distance()
        self._compute_approach()
        self.output_path = output_path

    @property
    def sessions(self):
        return list(sorted(set(self.data['ID'])))

    def _get_both_data(self, sess, re):
        features = ['timestamp', 'dist3d']
        for sub in ['b', 'm']:
            for pos in['X', 'Y', 'Z']:
                features.append('{}Pos3D_{}'.format(sub, pos))
            features.append('{}Approach'.format(sub))
        th = 2 if sess < 1000 else 3

        return self.data[(self.data['ID'] == sess) & (self.data['RE'] == re) &
                         (self.data['bInterpolationType'] <= th) & (self.data['mInterpolationType'] <= th)][features]

    def _get_infant_data(self, sess, re):
        features = ['timestamp', 'dist3d', 'bHeightFromFloor', 'bApproach']
        for pos in['X', 'Y', 'Z']:
            features.append('bPos3D_{}'.format(pos))
        th = 2 if sess < 1000 else 3

        return self.data[
            (self.data['ID'] == sess) & (self.data['RE'] == re) & (self.data['bInterpolationType'] <= th)
        ][features]

    def _get_mother_data(self, sess, re):
        features = ['timestamp', 'mApproach']
        for pos in['X', 'Y', 'Z']:
            features.append('mPos3D_{}'.format(pos))
        th = 2 if sess < 1000 else 3

        return self.data[
            (self.data['ID'] == sess) & (self.data['RE'] == re) & (self.data['mInterpolationType'] <= th)
        ][features]

    def _compute_distance(self):
        self.data['dist3d'] = np.sqrt(sum(
            (self.data['bPos3D_{}'.format(pos)] - self.data['mPos3D_{}'.format(pos)]) ** 2
            for pos in ['X', 'Y', 'Z']
        ))

    def _compute_approach(self):
        grouped = self.data.groupby(['ID', 'RE'])
        for gname, gdata in grouped:
            baby_movement = np.stack([
                gdata.iloc[1:]['bPos3D_{}'.format(pos)].to_numpy() -
                gdata.iloc[:-1]['bPos3D_{}'.format(pos)].to_numpy() for pos in ['X', 'Y', 'Z']
            ], axis=0)
            pos_diff = np.stack([
                gdata.iloc[1:]['mPos3D_{}'.format(pos)].to_numpy() -
                gdata.iloc[1:]['bPos3D_{}'.format(pos)].to_numpy() for pos in ['X', 'Y', 'Z']
            ], axis=0)
            self.data.loc[
                (self.data['ID'] == gname[0]) & (self.data['RE'] == gname[1]), 'bApproach'
            ] = np.concatenate((
                np.array([0]),
                np.sum(np.multiply(baby_movement, pos_diff), axis=0) / np.linalg.norm(pos_diff, axis=0) / 40.
            ))

            mother_movement = np.stack([
                gdata.iloc[1:]['mPos3D_{}'.format(pos)].to_numpy() -
                gdata.iloc[:-1]['mPos3D_{}'.format(pos)].to_numpy() for pos in ['X', 'Y', 'Z']
            ], axis=0)
            pos_diff = np.stack([
                gdata.iloc[1:]['bPos3D_{}'.format(pos)].to_numpy() -
                gdata.iloc[1:]['mPos3D_{}'.format(pos)].to_numpy() for pos in ['X', 'Y', 'Z']
            ], axis=0)
            self.data.loc[
                (self.data['ID'] == gname[0]) & (self.data['RE'] == gname[1]), 'mApproach'
            ] = np.concatenate((
                np.array([0]),
                np.sum(np.multiply(mother_movement, pos_diff), axis=0) / np.linalg.norm(pos_diff, axis=0) / 40.
            ))

    def _contact_duration(self, sess, re, th=DEFAULT_CONTACT_THRESHOLD):
        df = self._get_both_data(sess, re)
        return len(set(df[df['dist3d'] < th]['timestamp'])) * 0.04

    def _time_held(self, sess, re, th=DEFAULT_CARRY_THRESHOLD):
        df = self._get_infant_data(sess, re)
        return len(set(df[df['bHeightFromFloor'] > th]['timestamp'])) * 0.04

    def _velocity(self, sess, re, sub, direction):
        if sub == 'b':
            df = self._get_infant_data(sess, re)
        elif sub == 'm':
            df = self._get_mother_data(sess, re)
        else:
            raise NotImplemented

        if direction == 'toward':
            return np.mean(df[df['{}Approach'.format(sub)] > 0]['{}Approach'.format(sub)])
        elif direction == 'away':
            return np.mean(df[df['{}Approach'.format(sub)] < 0]['{}Approach'.format(sub)])
        else:
            raise NotImplemented

    def _initial_approach(self, sess, re, sub, th=DEFAULT_INIT_APP_TIME):
        if sub == 'b':
            df = self._get_infant_data(sess, re)
        elif sub == 'm':
            df = self._get_mother_data(sess, re)
        else:
            raise NotImplemented

        return np.mean(df[:th]['{}Approach'.format(sub)])

    def extract_features(self):
        features = defaultdict(list)

        for s in self.sessions:
            features['ID'].append(s)
            for r in [1, 2]:
                features['ContactDuration_R{}'.format(r)].append(self._contact_duration(s, r))
                features['TimeHeld_R{}'.format(r)].append(self._time_held(s, r))
                features['InfantInitialApproach_R{}'.format(r)].append(self._initial_approach(s, r, 'b'))
                features['InfantVelocityToward_R{}'.format(r)].append(self._velocity(s, r, 'b', 'toward'))
                features['InfantVelocityAway_R{}'.format(r)].append(self._velocity(s, r, 'b', 'away'))
                features['MotherInitialApproach_R{}'.format(r)].append(self._initial_approach(s, r, 'm'))
                features['MotherVelocityToward_R{}'.format(r)].append(self._velocity(s, r, 'm', 'toward'))
                features['MotherVelocityAway_R{}'.format(r)].append(self._velocity(s, r, 'm', 'away'))

        pd.DataFrame(features).to_csv(self.output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract Features from person tracking results')
    parser.add_argument('data_path', type=str, help='Path to the person tracking result data')
    parser.add_argument('output_path', type=str, help='Path to store the extracted features')
    args = parser.parse_args()
    data = AttachmentPersonTracking(Path(args.data_path), Path(args.output_path))
    data.extract_features()

