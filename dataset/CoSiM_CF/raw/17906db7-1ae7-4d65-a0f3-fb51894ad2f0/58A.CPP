#include <bits/stdc++.h>
using namespace std;

int main() {
  string s;
  cin >> s;
  bool can1 = 0, can2 = 0, can3 = 0, can4 = 0, can5 = 0;
  int h, e, l, l2;
  int size = s.size();
  for (int i = 0; i < size; i++) {
    if (s[i] == 'h') {
      h = i;
      can1 = 1;
      break;
    }
  }
  if (can1) {
    for (int j = h + 1; j < size; j++) {
      if (s[j] == 'e') {
        e = j;
        can2 = 1;
        break;
      }
    }
    if (can2) {
      for (int k = e + 1; k < size; k++) {
        if (s[k] == 'l') {
          l = k;
          can3 = 1;
          break;
        }
      }
      if (can3) {
        for (int x = l + 1; x < size; x++) {
          if (s[x] == 'l') {
            l2 = x;
            can4 = 1;
            break;
          }
        }
        if (can4) {
          for (int y = l2 + 1; y < size; y++) {
            if (s[y] == 'o') {
              can5 = 1;
              break;
            }
          }
        }
      }
    }
  }

  if (can1 && can2 && can3 && can4 && can5) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
}
