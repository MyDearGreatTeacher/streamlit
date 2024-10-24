# numpy1.py
```python
import streamlit as st
import time
import numpy as np
progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()
last_rows = np.random.randn(1, 1)
chart = st.line_chart(last_rows)
for i in range(1, 101):
    new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
    status_text.text("%i%% Complete" % i)
    chart.add_rows(new_rows)
    progress_bar.progress(i)
    last_rows = new_rows
    time.sleep(0.05)
progress_bar.empty()
# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
st.button("Re-run")
```

# numpy.py
```python
import streamlit as st
import numpy as np
binom_dist = np.random.binomial(1, .5, 100)
st.write(np.mean(binom_dist))
```
# numpy1.py
```python
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
binom_dist = np.random.binomial(1, .5, 1000)
list_of_means = []
for i in range(0, 1000):
     list_of_means.append(np.random.choice(binom_dist, 100, replace=True).mean())
fig, ax = plt.subplots()
ax = plt.hist(list_of_means)
st.pyplot(fig)
```
# numpy1.py
```python
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
binom_dist = np.random.binomial(1, .5, 1000)
list_of_means = []
for i in range(0, 1000):
     list_of_means.append(np.random.choice(binom_dist, 100, replace=True).mean())
plt.hist(list_of_means)
st.pyplot()
```
# numpy1.py
```python
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
binom_dist = np.random.binomial(1, .5, 1000)
list_of_means = []
for i in range(0, 1000):
     list_of_means.append(np.random.choice(binom_dist, 100, replace=True).mean())
plt.hist(list_of_means)
st.pyplot()
plt.hist([1,1,1,1])
st.pyplot()
```
# numpy1.py
```python
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
binom_dist = np.random.binomial(1, .5, 1000)
list_of_means = []
for i in range(0, 1000):
     list_of_means.append(np.random.choice(binom_dist, 100, replace=True).mean())
fig1, ax1 = plt.subplots()
ax1 = plt.hist(list_of_means)
st.pyplot(fig1)
fig2, ax2 = plt.subplots()
ax2 = plt.hist([1,1,1,1])
st.pyplot(fig2)
```
# numpy1.py
```python
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
perc_heads = st.number_input(label = 'Chance of Coins Landing on Heads', min_value = 0.0, max_value = 1.0, value = .5)
binom_dist = np.random.binomial(1, perc_heads, 1000)
list_of_means = []
for i in range(0, 1000):
     list_of_means.append(np.random.choice(binom_dist, 100, replace=True).mean())
fig, ax = plt.subplots()
ax = plt.hist(list_of_means, range=[0,1])
st.pyplot(fig)
```
# numpy1.py
```python
import streamlit as st  
import numpy as np  
import matplotlib.pyplot as plt
perc_heads = st.number_input(label='Chance of Coins Landing on Heads', min_value=0.0,  max_value=1.0, value=.5)
graph_title = st.text_input(label='Graph Title')
binom_dist = np.random.binomial(1, perc_heads, 1000)
list_of_means = []
for i in range(0, 1000):
list_of_means.append(np.random.choice(binom_dist, 100, replace=True).mean())
fig, ax = plt.subplots()
plt.hist(list_of_means, range=[0,1])
plt.title(graph_title)
st.pyplot(fig)
```
# numpy1.py
```python
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
st.title('Illustrating the Central Limit Theorem with Streamlit')
st.subheader('An App by Tyler Richards')
st.write(('This app simulates a thousand coin flips using the chance of heads input below,'
     'and then samples with replacement from that population and plots the histogram of the'
     ' means of the samples in order to illustrate the central limit theorem!'))
perc_heads = st.number_input(
    label='Chance of Coins Landing on Heads', min_value=0.0, max_value=1.0, value=.5)
binom_dist = np.random.binomial(1, perc_heads, 1000)
list_of_means = []
for i in range(0, 1000):
    list_of_means.append(np.random.choice(
        binom_dist, 100, replace=True).mean())
fig, ax = plt.subplots()
ax = plt.hist(list_of_means)
st.pyplot(fig)
```
# numpy1.py
```python

```
# numpy1.py
```python

```
# numpy1.py
```python

```
# numpy1.py
```python

```
# numpy1.py
```python

```
 
