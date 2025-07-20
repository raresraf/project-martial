#include <bits/stdc++.h>
using namespace std;
int main() {
  char str[100];
  int n, sum = 0;
  cin >> str;
  n = strlen(str);
  sort(str, str + n);
  for (int i = 0; i < n; i++) {
    if (str[i] != str[i + 1]) {
      sum++;
    }
  }
  if (sum % 2 == 0) {
    // cout<<sum;
    cout << "CHAT WITH HER!";
  } else {
    // cout<<sum;
    cout << "IGNORE HIM!";
  }
  return 0;
}
