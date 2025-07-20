#include <bits/stdc++.h>
using namespace std;

int main() {
  int len_s2;
  string s2, data_check;
  int found = 0;
  cin >> s2;
  len_s2 = s2.length();
  int count_lucky = 0;
  for (int i = 0; i < len_s2; i++) {
    if (s2[i] == '4' || s2[i] == '7') {
      count_lucky++;
    }
  }
  data_check = to_string(count_lucky);
  for (int i = 0; i < data_check.length(); i++) {
    if (data_check[i] != '4' && data_check[i] != '7') {
      found = 1;
    }
  }
  if (found == 0) {
    cout << "YES";
  } else {
    cout << "NO";
  }
  return 0;
}
