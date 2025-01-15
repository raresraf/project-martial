#include <bits/stdc++.h>

using namespace std;

int n;
string s[105];
char first;
char last;
int number;
char x[100];

int main() {
  int j = 0;
  int k = 0;

  cin >> n;

  for (int i = 0; i < n; i++) {
    cin >> s[i];
  }

  for (int i = 0; i < n; i++) {

    if (s[i].length() > 10) {
      // cout<<x[i];
      first = s[i][0];
      // cout<<first;
      last = s[i][s[i].length() - 1];
      // cout<<last;
      number = s[i].length() - 2;
      // cout<<number;

      j = 0;
      k = 0;
      while (number != 0) {
        x[j++] = '0' + number % 10;
        number /= 10;
      }

      while (j >= 0) {
        s[i][++k] = x[--j];
      }
      s[i][k++] = last;

      for (int p = k; p < s[i].length(); p++)
        s[i][p] = '\0';

      j = 0;
      k = 0;
    }
  }

  for (int i = 0; i < n; i++) {
    cout << s[i] << endl;
  }

  return 0;
}
