from scipy.io import arff
import pandas as pd
import numpy as np
data = arff.loadarff('src/assets/contact-lenses.arff')
df = pd.DataFrame(data[0])
x = np.array(df)
x = x.astype("U13")
d = data[1].names()
print(d)
