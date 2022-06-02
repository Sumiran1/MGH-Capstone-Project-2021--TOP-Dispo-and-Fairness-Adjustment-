# Can Artificial Intelligence Diagnose and Remedy, Instead of Consolidate, Racial Disparities in Care?

The use of Artificial Intelligence (AI) in clinical medicine risks perpetuating existing bias in care, such as disparities in access to post-injury rehabilitation services. We sought to leverage a novel, interpretable AI-based technology to: (1) uncover racial disparities in access to post-injury rehabilitation care and (2) create an AI-based prescriptive tool to address these disparities.

An interpretable AI methodology called Optimal Classification Trees (OCTs) was applied in an 80:20 derivation:validation split to predict discharge disposition (home vs. post-acute care [PAC]) in Black and White patients with penetrating injuries in the 2010-2016 ACS-TQIP database. The interpretable nature of OCTs allowed for examination of the AI logic to identify racial disparities in access to PAC. A prescriptive mixed-integer optimization model using age, injury, and gender data was allowed to "fairness-flip" the recommended discharge destination for a subset of patients while minimizing the ratio of imbalance between Black and White patients. The standard error or deviation of the injury severity, age, and gender variables was kept within a fixed tolerated range during the "fairness flip". Three OCTs were developed to predict discharge disposition: the first two trees used unadjusted data – one without and one with the race variable, and the third tree used fairness-adjusted data. Disparities and the discriminative performance (c-statistic) were compared among fairness-adjusted and unadjusted OCTs.

A total of 52,468 patients were included. The median age was 29 years, 12.0% were female, and 60.0% were Black; 21.5% of White patients and 12.1% of Black patients were discharged to PAC (p<0.001). Examining the AI logic uncovered significant disparities with race (Black vs. White) playing the second most important role in PAC discharge destination access. The prescriptive fairness adjustment recommended “flipping” the discharge destination of 4.5% of the patients, with the performance of the adjusted model increasing from a c-statistic (AUC) of 0.79 to 0.87. Following fairness adjustment, disparities disappeared and a similar percentage of Black and White patients (15.9% vs. 15.9%) had a recommended discharge to PAC.

We conclude that instead of accidentally encoding bias, interpretable AI methodologies are powerful tools to diagnose and remedy system-related bias in care, such as disparities in access to post-injury rehabilitation care.

Note:
 
The data for this project is too large to host on gtihub. Here is a link to the dropbox: , you will have to request write access so please feel free to request from the UI or email me at sumi123@berkeley.edu if you need access :)

Furthermore, besides the standard python and Julia libraries used, IAIs OCTs come with an IAI or interpretable AI license, so please keep that in mind.

 
