#include <bits/stdc++.h>
using namespace std;
int main() {
  int n;
  char ch1[101], ch2[101];
  cin >> ch1;
  cin >> ch2;
  n = strlen(ch1);
  for (int i = 0; i < n; i++) {
    char c = ch1[i];
    ch1[i] = tolower(c);
    ch2[i] = tolower(ch2[i]);
  }
  int flag = 0;
  int count = 0;
  for (int i = 0; i < n; i++) {
    if (ch1[i] > ch2[i]) {
      flag = 1;
      break;
    } else if (ch2[i] > ch1[i]) {
      flag = -1;
      break;
    }
  }
  cout << flag << '\n';
}
