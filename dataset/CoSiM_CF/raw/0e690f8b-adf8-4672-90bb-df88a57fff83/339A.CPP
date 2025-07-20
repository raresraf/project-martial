#include <bits/stdc++.h>
using namespace std;
int main() {
  string sum_numbers, output;
  int count_operator = 0, count_number = 0;

  cin >> sum_numbers;

  sort(sum_numbers.begin(), sum_numbers.end());

  cout << sum_numbers << endl;
  for (int i = 0; i < sum_numbers.length(); i++) {
    if (sum_numbers[i] == '+') {
      count_operator = count_operator + 1;
    } else {
      count_number = count_number + 1;
    }
  }
  cout << count_operator << endl;
  cout << count_number << endl;
  for (int i = count_operator; i < sum_numbers.length(); i = i + 1) {
    cout << sum_numbers[i];
    for (int j = count_operator; j < (sum_numbers.length() - 1); j = j + 1) {
      cout << '+';
      break;
    }
  }
  // cout<<output;
  return 0;
}
