import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats



##### DEMOGRAPHIC INFO & SUMMARY VALUES OF RAW DATA #####
def info_data(df):
    
    # count group size and gender distribution
    group_count = df['group'].value_counts()
    gender_count = df['gender'].value_counts()
    gender_group = df.groupby('group')['gender'].value_counts()
    
    # inspect age
    age_mean = df['age'].mean()
    age_std = df['age'].std()
    age_groupmean = df.groupby('group').age.mean()
    age_groupstd = df.groupby('group').age.std()
    age_stroke = df.loc[df.group.eq('stroke'), 'age']
    age_control = df.loc[df.group.eq('control'), 'age']
    age_stroke_distribution = stats.shapiro(age_stroke.values)
    age_control_distribution = stats.shapiro(age_control.values)
    age_variance = stats.levene(age_stroke.values, age_control.values)
    age_match = stats.ttest_ind(age_stroke.values, age_control.values)
    
    # print infos
    print('\n### DEMOGRAPHIC INFO ###')
    print('\nsubjects per group: '+str(group_count))
    print('\nsubjects per gender: '+str(gender_count))
    print('\nsubjects per group and gender: '+str(gender_group))
    print('\nage in stroke group: ' +str(age_stroke))
    print('\nage in control group: ' +str(age_control))
    print('\ntotal mean age: '+str(age_mean))
    print('\ntotal std of age: '+str(age_std))
    print('\nmean age by group: '+str(age_groupmean))
    print('\nstd of age by group: '+str(age_groupstd))
    print('\ndistribution of age in control subjects: '+str(age_control_distribution))
    print('distribution of age in stroke subjects: '+str(age_stroke_distribution))
    print('variance of age between groups: '+str(age_variance))
    print('t-test for age difference between groups: ' +str(age_match))

    
    # # mean values by group
    # mean = df.groupby('group').mean(numeric_only=True)
    
    # # mode values 
    # mode = df.groupby('group').agg(pd.Series.mode)
    # mode.drop(labels=['participant', 'TIME_start', 'TIME_end', 'country'], axis=1, inplace=True)


    # # median values
    # df2 = df.drop(labels=['age', 'TIME_start', 'TIME_end', 'TIME_total'], axis=1)
    # global df_median 
    # df_median = df2.groupby('group').median()
    # df_median.reset_index(inplace=True)
    
    # # print summary values
    # print('\nSUMMARIZATION VALUES')
    # print('\nmean: '+str(mean))
    # print('\nmode: ' +str(mode.to_string()))
    # print('\nmedian: ' + str(df_median))

    

##### REARRANGE DATA SET FOR FURTHER ANALYSIS & VISUALIZATION#####
def rearrange_data(df):
    
    # sort by group
    df.sort_values('group')
    
    # delete columns
    df2 = df.drop(labels=['TIME_start', 'TIME_end', 'TIME_total', 'gender', 'country'], axis=1)
    
    # merge settings into one column
    global df_melted1
    df_melted1 = df2.melt(id_vars=['participant', 'group', 'age'], var_name = 'env', value_name = 'value')
    df_melted1.sort_values('participant', inplace=True)

    # rearrange survey scores into environmemt, behaviour, emotion (columns)
    df_behaviour, df_emotion = df_melted1[(mask:=df_melted1['env'].str.contains('behaviour'))].copy(), df_melted1[~mask].copy()
    df_behaviour2 = df_behaviour.replace(to_replace=['behaviour_dom', 'behaviour_nature', 'behaviour_public', 'behaviour_traffic'], value=['domestic', 'nature', 'public', 'traffic'])
    df_behaviour2.rename(columns={'value': 'behaviour'}, inplace=True)
    df_emotion2 = df_emotion.replace(to_replace=['emotion_dom', 'emotion_nature', 'emotion_public', 'emotion_traffic'], value=['home', 'nature', 'public', 'traffic'])
    df_emotion2.rename(columns={'value': 'emotion'}, inplace=True)
    global df_melted2
    df_melted2 = df_behaviour2.merge(df_emotion2, how='inner', on=['env', 'participant', 'group', 'age'])
    df_melted2.sort_values('participant', inplace=True)
   
   

##### MANN-WHITNEY-U-TEST #####
def significance_test(df):
    # shapiro-wild test to check for normal distribution (-> p<0.05 means not normally distributed) 
    # + frequency histogram for visual check
    # mann-whitney-u test to check for significant difference between groups 
    # (requires non-normal distribution of underlying data)
    
    # prepare data frame
    df.drop(labels=['participant'], axis=1, inplace=True)
    df_stroke, df_control = df[(mask:=df['group'].str.contains('stroke'))].copy(), df[~mask].copy()
    df_stroke_beh, df_stroke_em = df_stroke[(mask:=df_stroke['env'].str.contains('behaviour'))].copy(), df_stroke[~mask].copy()
    df_control_beh, df_control_em = df_control[(mask:=df_control['env'].str.contains('behaviour'))].copy(), df_control[~mask].copy()
    
    #1: compare overall values of stroke and control
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('1: compare overall values of stroke vs. control')
    ax1.hist(df_stroke['value'], histtype='bar') 
    ax2.hist(df_control['value'],  histtype='bar') 
    ax1.set_xlabel('stroke')
    ax2.set_xlabel('control')
    plt.show()
    w1_1, pvalue1_1 = stats.shapiro(df_stroke['value']) #significance: not normally distributed
    w1_2, pvalue1_2 = stats.shapiro(df_control['value']) #significance: not normally distributed
    test1 = stats.mannwhitneyu(df_stroke['value'], df_control['value'], alternative='greater') #method='exact'
    print('\n### SIGNIFICANCE TESTING ###')
    print('\n1: compare overall values of stroke and control')
    print(w1_1, pvalue1_1)
    print(w1_2, pvalue1_2)
    print(test1)
    
    #2: compare behaviour and emotion in stroke
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('2: compare behaviour and emotion in stroke')
    ax1.hist(df_stroke_beh['value'], histtype='bar') 
    ax2.hist(df_stroke_em['value'],  histtype='bar') 
    ax1.set_xlabel('behaviour')
    ax2.set_xlabel('emotion')
    plt.show()
    w2_1, pvalue2_1 = stats.shapiro(df_stroke_beh['value']) #significance: not normally distributed
    w2_2, pvalue2_2 = stats.shapiro(df_stroke_em['value']) #significance: not normally distributed
    test2 = stats.mannwhitneyu(df_stroke_beh['value'], df_stroke_em['value'], alternative='two-sided') #method='exact'
    print('\n2: compare behaviour and emotion in stroke')
    print(w2_1, pvalue2_1)
    print(w2_2, pvalue2_2)
    print(test2)
    
    #3: compare behaviour and emotion in control
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('3: compare behaviour and emotion in control')
    ax1.hist(df_control_beh['value'], histtype='bar') 
    ax2.hist(df_control_em['value'],  histtype='bar') 
    ax1.set_xlabel('behaviour')
    ax2.set_xlabel('emotion')
    plt.show()
    w3_1, pvalue3_1 = stats.shapiro(df_control_beh['value']) #significance: not normally distributed
    w3_2, pvalue3_2 = stats.shapiro(df_control_em['value']) #significance: not normally distributed
    test3 = stats.mannwhitneyu(df_control_beh['value'], df_control_em['value'], alternative='two-sided') #method='exact'
    print('\n3: compare behaviour and emotion in control')
    print(w3_1, pvalue3_1)
    print(w3_2, pvalue3_2)
    print(test3)
    
    #4: compare behaviour in stroke and control
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('4: compare behaviour in stroke and control')
    ax1.hist(df_stroke_beh['value'], histtype='bar') 
    ax2.hist(df_control_beh['value'],  histtype='bar') 
    ax1.set_xlabel('stroke')
    ax2.set_xlabel('control')
    plt.show()
    w4_1, pvalue4_1 = stats.shapiro(df_stroke_beh['value']) #significance: not normally distributed
    w4_2, pvalue4_2 = stats.shapiro(df_control_beh['value']) #significance: not normally distributed
    test4 = stats.mannwhitneyu(df_stroke_beh['value'], df_control_beh['value'], alternative='greater') #method='exact'
    print('\n4: compare behaviour in stroke and control')
    print(w4_1, pvalue4_1)
    print(w4_2, pvalue4_2)
    print(test4)
    
    #5: compare emotion in stroke and control
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('5: compare emotion in stroke and control')
    ax1.hist(df_stroke_em['value'], histtype='bar') 
    ax2.hist(df_control_em['value'],  histtype='bar') 
    ax1.set_xlabel('stroke')
    ax2.set_xlabel('control')
    plt.show()
    w5_1, pvalue5_1 = stats.shapiro(df_stroke_em['value']) #significance: not normally distributed
    w5_2, pvalue5_2 = stats.shapiro(df_control_em['value']) #significance: not normally distributed
    test5 = stats.mannwhitneyu(df_stroke_em['value'], df_control_em['value'], alternative='greater') #method='exact'
    print('\n5: compare emotion in stroke and control')
    print(w5_1, pvalue5_1)
    print(w5_2, pvalue5_2)
    print(test5)
    
    #6: compare environments between stroke and control (domestic)
    df_stroke_dom = df_stroke[(mask:=df_stroke['env'].str.contains('dom'))].copy()
    df_control_dom = df_control[(mask:=df_control['env'].str.contains('dom'))].copy()
    df_stroke_dom_beh, df_stroke_dom_em = df_stroke_dom[(mask:=df_stroke_dom['env'].str.contains('beh'))].copy(), df_stroke_dom[~mask].copy()
    df_control_dom_beh, df_control_dom_em = df_control_dom[(mask:=df_control_dom['env'].str.contains('beh'))].copy(), df_control_dom[~mask].copy()
    #overall
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('6: compare environments between stroke and control (domestic overall)')
    ax1.hist(df_stroke_dom['value'], histtype='bar') 
    ax2.hist(df_control_dom['value'],  histtype='bar') 
    ax1.set_xlabel('stroke')
    ax2.set_xlabel('control')
    plt.show()
    w6_1_ov, pvalue6_1_ov = stats.shapiro(df_stroke_dom['value']) #significance: not normally distributed
    w6_2_ov, pvalue6_2_ov = stats.shapiro(df_control_dom['value']) #significance: not normally distributed
    test6_ov = stats.mannwhitneyu(df_stroke_dom['value'], df_control_dom['value'], alternative='greater', ) #method='exact'
    print('\n6: compare environments between stroke and control (domestic)\n(domestic overall)')
    print(w6_1_ov, pvalue6_1_ov)
    print(w6_2_ov, pvalue6_2_ov)
    print(test6_ov)
    #behaviour
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('6: compare environments between stroke and control (domestic behaviour)')
    ax1.hist(df_stroke_dom_beh['value'], histtype='bar') 
    ax2.hist(df_control_dom_beh['value'],  histtype='bar') 
    ax1.set_xlabel('stroke')
    ax2.set_xlabel('control')
    plt.show()
    w6_1_beh, pvalue6_1_beh = stats.shapiro(df_stroke_dom_beh['value'])
    w6_2_beh, pvalue6_2_beh = stats.shapiro(df_control_dom_beh['value'])
    test6_beh = stats.mannwhitneyu(df_stroke_dom_beh['value'], df_control_dom_beh['value'], alternative='greater') #method='exact'
    print('(domestic behaviour)')
    print(w6_1_beh, pvalue6_1_beh)
    print(w6_2_beh, pvalue6_2_beh)
    print(test6_beh)
    #emotion
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('6: compare environments between stroke and control (domestic emotion)')
    ax1.hist(df_stroke_dom_em['value'], histtype='bar') 
    ax2.hist(df_control_dom_em['value'],  histtype='bar') 
    ax1.set_xlabel('stroke')
    ax2.set_xlabel('control')
    plt.show()
    w6_1_em, pvalue6_1_em = stats.shapiro(df_stroke_dom_em['value']) #plot confirms non-normal distribution
    w6_2_em, pvalue6_2_em = stats.shapiro(df_control_dom_em['value']) #significance: not normally distributed
    test6_em = stats.mannwhitneyu(df_stroke_dom_em['value'], df_control_dom_em['value'], alternative='greater') #method='exact'
    print('(domestic emotion)')
    print(w6_1_em, pvalue6_1_em)
    print(w6_2_em, pvalue6_2_em)
    print(test6_em)
    
    #7: compare environments between stroke and control (nature)
    df_stroke_nat = df_stroke[(mask:=df_stroke['env'].str.contains('nature'))].copy()
    df_control_nat = df_control[(mask:=df_control['env'].str.contains('nature'))].copy()
    df_stroke_nat_beh, df_stroke_nat_em = df_stroke_nat[(mask:=df_stroke_nat['env'].str.contains('beh'))].copy(), df_stroke_nat[~mask].copy()
    df_control_nat_beh, df_control_nat_em = df_control_nat[(mask:=df_control_nat['env'].str.contains('beh'))].copy(), df_control_nat[~mask].copy()
    #overall
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('7: compare environments between stroke and control (nature overall)')
    ax1.hist(df_stroke_nat['value'], histtype='bar') 
    ax2.hist(df_control_nat['value'],  histtype='bar') 
    ax1.set_xlabel('stroke')
    ax2.set_xlabel('control')
    plt.show()
    w7_1_ov, pvalue7_1_ov = stats.shapiro(df_stroke_nat['value']) #no significance
    w7_2_ov, pvalue7_2_ov = stats.shapiro(df_control_nat['value']) #significance: not normally distributed
    test7_ov = stats.mannwhitneyu(df_stroke_nat['value'], df_control_nat['value'], alternative='greater') #method='exact'
    print('\n7: compare environments between stroke and control (nature)\n(nature overall)')
    print(w7_1_ov, pvalue7_1_ov)
    print(w7_2_ov, pvalue7_2_ov)
    print(test7_ov)
    #behaviour
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('7: compare environments between stroke and control (nature behaviour)')
    ax1.hist(df_stroke_nat_beh['value'], histtype='bar') 
    ax2.hist(df_control_nat_beh['value'],  histtype='bar') 
    ax1.set_xlabel('stroke')
    ax2.set_xlabel('control')
    plt.show()
    w7_1_beh, pvalue7_1_beh = stats.shapiro(df_stroke_nat_beh['value']) #significance: not normally distributed
    w7_2_beh, pvalue7_2_beh = stats.shapiro(df_control_nat_beh['value']) #significance: not normally distributed
    test7_beh = stats.mannwhitneyu(df_stroke_nat_beh['value'], df_control_nat_beh['value'], alternative='greater') #method='exact'
    print('(nature behaviour)')
    print(w7_1_beh, pvalue7_1_beh)
    print(w7_2_beh, pvalue7_2_beh)
    print(test7_beh)
    #emotion
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('7: compare environments between stroke and control (nature emotion)')
    ax1.hist(df_stroke_nat_em['value'], histtype='bar') 
    ax2.hist(df_control_nat_em['value'],  histtype='bar') 
    ax1.set_xlabel('stroke')
    ax2.set_xlabel('control')
    plt.show()
    w7_1_em, pvalue7_1_em = stats.shapiro(df_stroke_nat_em['value']) #significance: not normally distributed
    w7_2_em, pvalue7_2_em = stats.shapiro(df_control_nat_em['value']) #significance: not normally distributed
    test7_em = stats.mannwhitneyu(df_stroke_nat_em['value'], df_control_nat_em['value'], alternative='greater') #method='exact'
    print('(nature emotion)')
    print(w7_1_em, pvalue7_1_em)
    print(w7_2_em, pvalue7_2_em)
    print(test7_em)
    
    #8: compare environments between stroke and control (public)
    df_stroke_pub = df_stroke[(mask:=df_stroke['env'].str.contains('public'))].copy()
    df_control_pub = df_control[(mask:=df_control['env'].str.contains('public'))].copy()
    df_stroke_pub_beh, df_stroke_pub_em = df_stroke_pub[(mask:=df_stroke_pub['env'].str.contains('beh'))].copy(), df_stroke_pub[~mask].copy()
    df_control_pub_beh, df_control_pub_em = df_control_pub[(mask:=df_control_pub['env'].str.contains('beh'))].copy(), df_control_pub[~mask].copy()
    #overall
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('8: compare environments between stroke and control (public overall)')
    ax1.hist(df_stroke_pub['value'], histtype='bar') 
    ax2.hist(df_control_pub['value'],  histtype='bar') 
    ax1.set_xlabel('stroke')
    ax2.set_xlabel('control')
    plt.show()
    w8_1_ov, pvalue8_1_ov = stats.shapiro(df_stroke_pub['value']) #significance: not normally distributed
    w8_2_ov, pvalue8_2_ov = stats.shapiro(df_control_pub['value']) #significance: not normally distributed
    test8_ov = stats.mannwhitneyu(df_stroke_pub['value'], df_control_pub['value'], alternative='greater') #method='exact'
    print('\n8: compare environments between stroke and control (public)\n(public overall)')
    print(w8_1_ov, pvalue8_1_ov)
    print(w8_2_ov, pvalue8_2_ov)
    print(test8_ov)
    #behaviour
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('8: compare environments between stroke and control (public behaviour)')
    ax1.hist(df_stroke_pub_beh['value'], histtype='bar') 
    ax2.hist(df_control_pub_beh['value'],  histtype='bar') 
    ax1.set_xlabel('stroke')
    ax2.set_xlabel('control')
    plt.show()
    w8_1_beh, pvalue8_1_beh = stats.shapiro(df_stroke_nat_beh['value']) #significance: not normally distributed
    w8_2_beh, pvalue8_2_beh = stats.shapiro(df_control_nat_beh['value']) #significance: not normally distributed
    test8_beh = stats.mannwhitneyu(df_stroke_nat_beh['value'], df_control_nat_beh['value'], alternative='greater') #method='exact'
    print('(public behaviour)')
    print(w8_1_beh, pvalue8_1_beh)
    print(w8_2_beh, pvalue8_2_beh)
    print(test8_beh)
    #emotion
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('8: compare environments between stroke and control (public emotion)')
    ax1.hist(df_stroke_pub_em['value'], histtype='bar') 
    ax2.hist(df_control_pub_em['value'],  histtype='bar') 
    ax1.set_xlabel('stroke')
    ax2.set_xlabel('control')
    plt.show()
    w8_1_em, pvalue8_1_em = stats.shapiro(df_stroke_pub_em['value']) #significance: not normally distributed
    w8_2_em, pvalue8_2_em = stats.shapiro(df_control_pub_em['value']) #plot confirms non-normal distribution
    test8_em = stats.mannwhitneyu(df_stroke_pub_em['value'], df_control_pub_em['value'], alternative='greater') #method='exact'
    print('public emotion)')
    print(w8_1_em, pvalue8_1_em)
    print(w8_2_em, pvalue8_2_em)
    print(test8_em)
    
    #9: compare environments between stroke and control (traffic)
    df_stroke_traf = df_stroke[(mask:=df_stroke['env'].str.contains('traffic'))].copy()
    df_control_traf = df_control[(mask:=df_control['env'].str.contains('traffic'))].copy()
    df_stroke_traf_beh, df_stroke_traf_em = df_stroke_traf[(mask:=df_stroke_traf['env'].str.contains('beh'))].copy(), df_stroke_traf[~mask].copy()
    df_control_traf_beh, df_control_traf_em = df_control_traf[(mask:=df_control_traf['env'].str.contains('beh'))].copy(), df_control_traf[~mask].copy()
    #overall
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('9: compare environments between stroke and control (traffic overall)')
    ax1.hist(df_stroke_traf['value'], histtype='bar') 
    ax2.hist(df_control_traf['value'],  histtype='bar') 
    ax1.set_xlabel('stroke')
    ax2.set_xlabel('control')
    plt.show()
    w9_1_ov, pvalue9_1_ov = stats.shapiro(df_stroke_traf['value']) #no significance
    w9_2_ov, pvalue9_2_ov = stats.shapiro(df_control_traf['value']) #significance: not normally distributed
    test9_ov = stats.mannwhitneyu(df_stroke_traf['value'], df_control_traf['value'], alternative='greater') #method='exact'
    print('\n9: compare environments between stroke and control (traffic)\n(traffic overall)')
    print(w9_1_ov, pvalue9_1_ov)
    print(w9_2_ov, pvalue9_2_ov)
    print(test9_ov)
    #behaviour
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('9: compare environments between stroke and control (traffic behaviour)')
    ax1.hist(df_stroke_traf_beh['value'], histtype='bar') 
    ax2.hist(df_control_traf_beh['value'],  histtype='bar') 
    ax1.set_xlabel('stroke')
    ax2.set_xlabel('control')
    plt.show()
    w9_1_beh, pvalue9_1_beh = stats.shapiro(df_stroke_traf_beh['value']) #plot confirms non-normal distribution
    w9_2_beh, pvalue9_2_beh = stats.shapiro(df_control_traf_beh['value']) #significance: not normally distributed
    test9_beh = stats.mannwhitneyu(df_stroke_traf_beh['value'], df_control_traf_beh['value'], alternative='greater') #method='exact'
    print('(traffic behaviour)')
    print(w9_1_beh, pvalue9_1_beh)
    print(w9_2_beh, pvalue9_2_beh)
    print(test9_beh)
    #emotion
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('9: compare environments between stroke and control (traffic emotion)')
    ax1.hist(df_stroke_traf_em['value'], histtype='bar') 
    ax2.hist(df_control_traf_em['value'],  histtype='bar') 
    ax1.set_xlabel('stroke')
    ax2.set_xlabel('control')
    plt.show()
    w9_1_em, pvalue9_1_em = stats.shapiro(df_stroke_traf_em['value']) #plot confirms non-normal distribution
    w9_2_em, pvalue9_2_em = stats.shapiro(df_control_traf_em['value']) #significance: not normally distributed
    test9_em = stats.mannwhitneyu(df_stroke_traf_em['value'], df_control_traf_em['value'], alternative='greater') #method='exact'
    print('(traffic emotion)')
    print(w9_1_em, pvalue9_1_em)
    print(w9_2_em, pvalue9_2_em)
    print(test9_em)
    
    #10: compare environments within groups (stroke)
    #overall
    test10_1_ov = stats.mannwhitneyu(df_stroke_dom['value'], df_stroke_nat['value'])
    test10_2_ov = stats.mannwhitneyu(df_stroke_dom['value'], df_stroke_pub['value'])
    test10_3_ov = stats.mannwhitneyu(df_stroke_dom['value'], df_stroke_traf['value'])
    test10_4_ov = stats.mannwhitneyu(df_stroke_nat['value'], df_stroke_pub['value'])
    test10_5_ov = stats.mannwhitneyu(df_stroke_nat['value'], df_stroke_traf['value'])
    test10_6_ov = stats.mannwhitneyu(df_stroke_pub['value'], df_stroke_traf['value'])
    print('\n10: compare environments within groups (stroke)\n(overall)')
    print('dom vs nat: ' + str(test10_1_ov))
    print('dom vs pub: ' + str(test10_2_ov))
    print('dom vs traf: ' + str(test10_3_ov))
    print('nat vs pub: ' + str(test10_4_ov))
    print('nat vs traf: ' + str(test10_5_ov))
    print('pub vs traf: ' + str(test10_6_ov))
    #behaviour
    test10_1_beh = stats.mannwhitneyu(df_stroke_dom_beh['value'], df_stroke_nat_beh['value'])
    test10_2_beh = stats.mannwhitneyu(df_stroke_dom_beh['value'], df_stroke_pub_beh['value'])
    test10_3_beh = stats.mannwhitneyu(df_stroke_dom_beh['value'], df_stroke_traf_beh['value'])
    test10_4_beh = stats.mannwhitneyu(df_stroke_nat_beh['value'], df_stroke_pub_beh['value'])
    test10_5_beh = stats.mannwhitneyu(df_stroke_nat_beh['value'], df_stroke_traf_beh['value'])
    test10_6_beh = stats.mannwhitneyu(df_stroke_pub_beh['value'], df_stroke_traf_beh['value'])
    print('(behaviour)')
    print('dom vs nat: ' + str(test10_1_beh))
    print('dom vs pub: ' + str(test10_2_beh))
    print('dom vs traf: ' + str(test10_3_beh))
    print('nat vs pub: ' + str(test10_4_beh))
    print('nat vs traf: ' + str(test10_5_beh))
    print('pub vs traf: ' + str(test10_6_beh))
    #emotion
    test10_1_em = stats.mannwhitneyu(df_stroke_dom_em['value'], df_stroke_nat_em['value'])
    test10_2_em = stats.mannwhitneyu(df_stroke_dom_em['value'], df_stroke_pub_em['value'])
    test10_3_em = stats.mannwhitneyu(df_stroke_dom_em['value'], df_stroke_traf_em['value'])
    test10_4_em = stats.mannwhitneyu(df_stroke_nat_em['value'], df_stroke_pub_em['value'])
    test10_5_em = stats.mannwhitneyu(df_stroke_nat_em['value'], df_stroke_traf_em['value'])
    test10_6_em = stats.mannwhitneyu(df_stroke_pub_em['value'], df_stroke_traf_em['value'])
    print('(emotion)')
    print('dom vs nat: ' + str(test10_1_em))
    print('dom vs pub: ' + str(test10_2_em))
    print('dom vs traf: ' + str(test10_3_em))
    print('nat vs pub: ' + str(test10_4_em))
    print('nat vs traf: ' + str(test10_5_em))
    print('pub vs traf: ' + str(test10_6_em))
    
    #11: compare environments within groups (control)
    #overall
    test11_1_ov = stats.mannwhitneyu(df_control_dom['value'], df_control_nat['value'])
    test11_2_ov = stats.mannwhitneyu(df_control_dom['value'], df_control_pub['value'])
    test11_3_ov = stats.mannwhitneyu(df_control_dom['value'], df_control_traf['value'])
    test11_4_ov = stats.mannwhitneyu(df_control_nat['value'], df_control_pub['value'])
    test11_5_ov = stats.mannwhitneyu(df_control_nat['value'], df_control_traf['value'])
    test11_6_ov = stats.mannwhitneyu(df_control_pub['value'], df_control_traf['value'])
    print('\n11: compare environments within groups (control)\n(overall)')
    print('dom vs nat: ' + str(test11_1_ov))
    print('dom vs pub: ' + str(test11_2_ov))
    print('dom vs traf: ' + str(test11_3_ov))
    print('nat vs pub: ' + str(test11_4_ov))
    print('nat vs traf: ' + str(test11_5_ov))
    print('pub vs traf: ' + str(test11_6_ov))
    #behaviour
    test11_1_beh = stats.mannwhitneyu(df_control_dom_beh['value'], df_control_nat_beh['value'])
    test11_2_beh = stats.mannwhitneyu(df_control_dom_beh['value'], df_control_pub_beh['value'])
    test11_3_beh = stats.mannwhitneyu(df_control_dom_beh['value'], df_control_traf_beh['value'])
    test11_4_beh = stats.mannwhitneyu(df_control_nat_beh['value'], df_control_pub_beh['value'])
    test11_5_beh = stats.mannwhitneyu(df_control_nat_beh['value'], df_control_traf_beh['value'])
    test11_6_beh = stats.mannwhitneyu(df_control_pub_beh['value'], df_control_traf_beh['value'])
    print('(behaviour)')
    print('dom vs nat: ' + str(test11_1_beh))
    print('dom vs pub: ' + str(test11_2_beh))
    print('dom vs traf: ' + str(test11_3_beh))
    print('nat vs pub: ' + str(test11_4_beh))
    print('nat vs traf: ' + str(test11_5_beh))
    print('pub vs traf: ' + str(test11_6_beh))
    #emotion
    test11_1_em = stats.mannwhitneyu(df_control_dom_em['value'], df_control_nat_em['value'])
    test11_2_em = stats.mannwhitneyu(df_control_dom_em['value'], df_control_pub_em['value'])
    test11_3_em = stats.mannwhitneyu(df_control_dom_em['value'], df_control_traf_em['value'])
    test11_4_em = stats.mannwhitneyu(df_control_nat_em['value'], df_control_pub_em['value'])
    test11_5_em = stats.mannwhitneyu(df_control_nat_em['value'], df_control_traf_em['value'])
    test11_6_em = stats.mannwhitneyu(df_control_pub_em['value'], df_control_traf_em['value'])
    print('(emotion)')
    print('dom vs nat: ' + str(test11_1_em))
    print('dom vs pub: ' + str(test11_2_em))
    print('dom vs traf: ' + str(test11_3_em))
    print('nat vs pub: ' + str(test11_4_em))
    print('nat vs traf: ' + str(test11_5_em))
    print('pub vs traf: ' + str(test11_6_em))
    


##### CORRELATION AGE - HAZARD PERCEPTION #####   
def spearman_correlation(df_input):
    
    # prepare data frame
    df = df_input.copy()
    df['total'] = df['behaviour'] + df['emotion']
    df_stroke, df_control = df[(mask:=df['group'].str.contains('stroke'))].copy(), df[~mask].copy()
    
    all_total = stats.spearmanr(df['age'], df['total']) #negligible
    all_behaviour = stats.spearmanr(df['age'], df['behaviour']) #negligible
    all_emotion = stats.spearmanr(df['age'], df['emotion']) #negligible
    stroke_total = stats.spearmanr(df_stroke['age'], df_stroke['total']) #negligible
    stroke_behaviour = stats.spearmanr(df_stroke['age'], df_stroke['behaviour']) #weak
    stroke_emotion = stats.spearmanr(df_stroke['age'], df_stroke['emotion']) #moderate
    control_total = stats.spearmanr(df_control['age'], df_control['total']) #negligible
    control_behaviour = stats.spearmanr(df_control['age'], df_control['behaviour']) #negligible
    control_emotion = stats.spearmanr(df_control['age'], df_control['emotion']) #negligible
   
    print('\n### CORRELATION AGE & HAZARD PERCEPTION ###')
    print('\nall:')
    print('(total)\n ' + str(all_total))
    print('(behaviour)\n' + str(all_behaviour))
    print('(emotion)\n' + str(all_emotion))
    print('\nstroke:')
    print('(total)\n ' + str(stroke_total))
    print('(behaviour)\n' + str(stroke_behaviour))
    print('(emotion)\n' + str(stroke_emotion))
    print('\ncontrol:')
    print('(total)\n ' + str(control_total))
    print('(behaviour)\n' + str(control_behaviour))
    print('(emotion)\n' + str(control_emotion))
    
    
    
#####TWO-SIDED BAR CHART: MEDIAN SINGLE VALUES STROKE VS. CONTROL #####
def plot_twosided_bar(df):
    
    # prepare data frame
    df_melted = df.melt(id_vars=['group'],
                    var_name = 'env',
                    value_name = 'value')

    
    df_melted_behaviour, df_melted_emotion = df_melted[(mask:=df_melted.env.str.contains('behaviour'))].copy(), df_melted[~mask].copy()
    
    df_med_behaviour = df_melted_behaviour.replace(to_replace=['behaviour_dom', 'behaviour_nature', 'behaviour_public', 'behaviour_traffic'], value=['home', 'nature', 'public', 'traffic'])
    df_med_behaviour.rename(columns={'value': 'behaviour'}, inplace=True)
    
    df_med_emotion = df_melted_emotion.replace(to_replace=['emotion_dom', 'emotion_nature', 'emotion_public', 'emotion_traffic'], value=['home', 'nature', 'public', 'traffic'])
    df_med_emotion.rename(columns={'value': 'emotion'}, inplace=True)
    
    df_med_sorted = df_med_behaviour.merge(df_med_emotion, how='inner', on=['group', 'env'])
    df_med_sorted.sort_values(by='group', inplace=True)
    
    # plot
    df_med_sorted.loc[df_med_sorted.group.eq('stroke'), 'behaviour'] = df_med_sorted['behaviour'].mul(-1)
    df_med_sorted.loc[df_med_sorted.group.eq('stroke'), 'emotion'] = df_med_sorted['emotion'].mul(-1)
    df_stroke, df_control = df_med_sorted[(mask:=df_med_sorted['group'].str.contains('stroke'))].copy(), df_med_sorted[~mask].copy()
    df_stroke.set_index(['env'])
    df_control.set_index(['env'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,7), sharey=True)
    df_stroke.set_index(['env']).loc[['traffic', 'public', 'nature', 'home']].plot.barh(ax=ax1, legend=False, color=['tab:blue', 'darkorange'])
    df_control.set_index(['env']).loc[['traffic', 'public', 'nature', 'home']].plot.barh(ax=ax2, color=['tab:blue', 'darkorange'])
    plt.style.use('bmh')
    
    ax1.set_title('Stroke', pad=12, weight='bold')
    ax1.yaxis.label.set_visible(False)
    ax1.yaxis.set_ticks_position('right')
    ax1.tick_params(which='both', right=False, bottom=False)
    ax1.xaxis.set_ticks(np.arange(-4, 0, 1), ['4', '3', '2', '1'])
    ax1.grid(visible=True, axis='x', which='major', color='dimgray', linestyle='-')
    ax1.grid(visible=False, axis='y', which='both')
    

    ax2.set_title('Control', pad=12, weight='bold')
    ax2.tick_params(which='both', left=False, bottom=False)
    ax2.xaxis.set_ticks(np.arange(1, 5, 1), ['1', '2', '3', '4'])
    ax2.yaxis.set_ticks(np.arange(0, 4, 1), ['Traffic', 'Public', 'Nature', 'Home'])
    ax2.grid(visible=True, axis='x', which='major', color='dimgray', linestyle='-')
    ax2.grid(visible=False, axis='y', which='both')
    pos = ax2.get_position()
    ax2.set_position([pos.x0, pos.y0, pos.width*1, pos.height])
    ax2.legend(loc='center right', bbox_to_anchor=(0.25, 1.035), ncol=2, frameon=False, labels=['Behaviour', 'Emotion'])

    plt.subplots_adjust(wspace=0.12)
    plt.subplots_adjust(top=0.85) 

    plt.show()
    


##### MEAN RISKPERCEPTION STROKE VS. CONTROL#####
def plot_riskperception(df):

    # prepare data frame
    df.drop(labels=['age'], axis=1, inplace=True)
    df_mean = df.groupby('group').mean()
    df_mean.sort_values('group', ascending=False, inplace=True)
    df_std = df.groupby(['group'])['behaviour', 'emotion'].std()
    df_total = df.copy()
    df_total['total'] = df_total['behaviour'] + df_total['emotion']
    df_total.drop(columns=['behaviour', 'emotion'], inplace=True)
    df_total_mean = df_total.groupby('group').mean()
    df_total_mean.sort_values('group', ascending=False, inplace=True)
    df_total_mean['percentage'] = ((df_total_mean['total']/8)*100).round(2)
    df_total_perc = df_total_mean.drop(columns=['total'])
    
    # plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7))
    df_total_perc.plot.bar(ax=ax1, color='dimgray') #yerr=df_total_std, error_kw=dict(ecolor='dimgray', lw=2, capsize=5, capthick=2), 
    ax1.legend(loc='upper center', ncol=1, frameon=True, labels=['Total in %'])
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('top')
    ax1.set_xlabel(None)
    ax1.set_yticks([0,25,50,75,100], ['','25%','50%','75%','100%'])
    ax1.bar_label(ax1.containers[0])
    ax1.tick_params(top=False, left=False)
    ax1.grid(axis='y', which='major', color='dimgray', linestyle='-')
    ax1.xaxis.set_ticks([0,1], labels=['Stroke', 'Control'], rotation=0, weight='bold', fontsize=12)
    ax1.grid(visible=False, axis='x', which='both')
    
    df_mean.plot.bar(ax=ax2, yerr=df_std, error_kw=dict(ecolor='dimgray', lw=2, capsize=5, capthick=2), color=['tab:blue', 'darkorange'])
    ax2.set_xlabel(None)
    ax2.yaxis.set_ticks_position('right')
    ax2.xaxis.set_ticks_position('top')
    ax2.xaxis.set_ticks([0,1], labels=['Stroke', 'Control'], rotation=0, weight='bold', fontsize=12)
    ax2.set_yticks([0,1,2,3,4], ['','1','2','3','4'])
    ax2.grid(axis='y', which='major', color='dimgray', linestyle='-')
    ax2.legend(loc='upper center', ncol=1, frameon=True, labels=['Behaviour', 'Emotion'])
    ax2.tick_params(top=False, which='both', right=False)
    ax2.grid(visible=False, axis='x', which='both')

    plt.subplots_adjust(wspace=0.03)

    plt.show()
    

    
##### MAIN & PREPROCESSING #####
def main():
    
    # input csv-data file
    df = pd.read_csv('C:/Users/hgaud/OneDrive/Studium/8. SoSe2022/BA/Questionnaire/data-2022-07-20/data.csv', index_col=False)
    
    # rename columns 
    df.columns = ['participant', 'age',	'gender', 'group', 'behaviour_dom', 'behaviour_nature',	'behaviour_public',	'behaviour_traffic',
                  'emotion_dom', 'emotion_nature', 'emotion_public', 'emotion_traffic', 'country', 'TIME_start', 'TIME_end', 'TIME_total']
   
    #remove incomplete trials
    incomplete = df.isnull().any(axis=1).sum()  
    df.dropna(axis=0, how = 'any', inplace=True)
    print('incomplete trials (removed): '+str(incomplete))
    
    #remove subjects younger than 60 years old
    young = (df.age < 60).sum()
    df = df[df.age >= 60]
    print('subjects younger than 60 years old (removed): '+str(young))
    
    # replace number coding of gender and group with actual strings
    df['gender'].replace({1:'female', 2:'male', 3:'other'}, inplace=True)
    df['group'].replace({1:'stroke', 2:'control'}, inplace=True)

    
    # demographic info & summary values of raw data
    first_infos = info_data(df)
    print(first_infos)
    
    # #rearrange data set for further analysis & visualization
    # rearrange = rearrange_data(df)
    # print(rearrange)
    
    # # significance testing (Shapiro-Wilk-Test, Mann-Withney-U-Test)
    # test = significance_test(df_melted1)
    # print(test)
    
    # # correlation age - hazard perception (Spearman rank-order correlation)
    # correlation = spearman_correlation(df_melted2)
    # print(correlation)
    
    # # visualization of medians of single variables stroke vs. control
    # twosided_bar = plot_twosided_bar(df_median)
    # print(twosided_bar)
    
    # # visualization of mean behavioural, emotional & total riskperception stroke vs. control
    # riskperception = plot_riskperception(df_melted2)
    # print(riskperception)

if __name__ == '__main__':
    main()
