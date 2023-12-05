import numpy as np
import pandas as pd

# CSV 파일 읽기
data = pd.read_csv('2019_kbo_for_kaggle_v2.csv')

for i in range(2015, 2019):
    # 'year' 열이 i인 행만 선택
    filtered_data = data[data['year'] == i]

    # 안타개수를 오름차순으로 정렬하되 10개
    print(f"{i} 안타개수 상위 10명")
    top_10_H = filtered_data.sort_values(by='H', ascending=False).head(10)
    result_hits = top_10_H['batter_name']
    print(result_hits)

    print(f"{i} 타율 상위 10명")
    top_10_avg = filtered_data.sort_values(by='avg', ascending=False).head(10)
    result_avg = top_10_avg['batter_name']
    print(result_avg)

    print(f"{i} 홈런 상위 10명")
    top_10_HR = filtered_data.sort_values(by='HR', ascending=False).head(10)
    result_hr = top_10_HR['batter_name']
    print(result_hr)

    print(f"{i} 출루율 상위 10명")
    top_10_OBP = filtered_data.sort_values(by='OBP', ascending=False).head(10)
    result_obp = top_10_OBP['batter_name']
    print(result_obp)

for position in ['포수', '1루수', '2루수', '3루수', '유격수', '좌익수', '중견수', '우익수']:
    print(f"2018년도 가장 높은 {position} war")
    selected_position = data[(data['year'] == 2018) & (data['cp'] == position)]
    top1_position_row = selected_position.sort_values(by='war', ascending=False).head(1)
    top1_position_name = top1_position_row['batter_name']
    print(top1_position_name)

for stat in ['R', 'H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG']:
    print(f"{stat}와 연봉의 상관관계")
    print(data['salary'].corr(data[stat]))

print("연봉과 가장 상관관계가 높은 수치는 RBI(타점) 입니다.")
