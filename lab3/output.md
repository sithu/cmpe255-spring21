| Experiment | Accuracy           | Confusion Matrix      | Comment                                                                           |
|------------|--------------------|-----------------------|-----------------------------------------------------------------------------------|
| Baseline   | 0.6770833333333334 | [[114  16] [ 46  16]] |  |
| Solution 1 | 0.6770833333333334 | [[115, 15], [47, 15]] | Used features:  ['skin', 'insulin', 'bmi', 'age'] 
| Solution 2 | 0.7708333333333334 | [[115, 15], [29, 33]] | Used features:  ['glucose', 'insulin', 'bmi', 'age'] 
| Solution 3 | 0.7864583333333334 | [[117, 13], [28, 34]] | Used features:  ['pregnant', 'glucose', 'insulin', 'bmi', 'age'] 
| Solution 4 | 0.8020833333333334 | [[118, 12], [26, 36]] | Used features:  ['pregnant', 'glucose', 'bp', 'insulin', 'bmi', 'age', 'pedigree'] 