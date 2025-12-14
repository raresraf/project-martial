#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, t;
  cin >> n >> t;
  string inp;
  cin >> inp;

  for (int i = 0; i < t; i++) {
    string temp = inp;
    for (int j = 0; j < n - 1; j++) {
      if (inp[j] == 'B' && inp[j + 1] == 'G') {
        temp[j] = 'G';
        temp[j + 1] = 'B';
      }
    }
    inp = temp;
  }

  cout << inp;
}
