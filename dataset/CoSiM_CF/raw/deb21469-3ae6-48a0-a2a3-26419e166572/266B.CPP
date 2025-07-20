#include <bits/stdc++.h>
using namespace std;
int main() {
  ios::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);
  string str;
  int siz;
  cin >> siz;
  int n;
  cin >> n;
  cin >> str;
  while (n--) {
    for (int i = 0; i < siz; i++) {
      if (str[i] == 'B' && str[i + 1] == 'G') {
        swap(str[i], str[i + 1]);
        i++;
      }
    }
  }
  cout << str;
  return 0;
}
