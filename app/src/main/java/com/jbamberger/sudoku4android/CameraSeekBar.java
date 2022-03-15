package com.jbamberger.sudoku4android;


import android.widget.SeekBar;
import android.widget.TextView;

class CameraSeekBar {
    int _progress;
    long _min, _max, _absVal;
    SeekBar _seekBar;
    TextView _sliderPrompt;

    CameraSeekBar() {
        _progress = 0;
        _min = _max = _absVal = 0;
    }

    CameraSeekBar(SeekBar seekBar, TextView textView, long min, long max, long val) {
        _seekBar = seekBar;
        _sliderPrompt = textView;
        _min = min;
        _max = max;
        _absVal = val;

        if (_min != _max) {
            _progress = (int) ((_absVal - _min) * _seekBar.getMax() / (_max - _min));
            seekBar.setProgress(_progress);
            updateProgress(_progress);
        } else {
            _progress = 0;
            seekBar.setEnabled(false);
        }
    }

    public boolean isSupported() {
        return (_min != _max);
    }

    public void updateProgress(int progress) {
        if (!isSupported())
            return;

        _progress = progress;
        _absVal = (progress * (_max - _min)) / _seekBar.getMax() + _min;
        int val = (progress * (_seekBar.getWidth() - 2 * _seekBar.getThumbOffset())) / _seekBar.getMax();
        _sliderPrompt.setText("" + _absVal);
        _sliderPrompt.setX(_seekBar.getX() + val + _seekBar.getThumbOffset() / 2);
    }

    public int getProgress() {
        return _progress;
    }

    public void updateAbsProgress(long val) {
        if (!isSupported())
            return;
        int progress = (int) ((val - _min) * _seekBar.getMax() / (_max - _min));
        updateProgress(progress);
    }

    public long getAbsProgress() {
        return _absVal;
    }
}

