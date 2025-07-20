#include <bits/stdc++.h>
using namespace std;

void yoyo(int &a, int &b) {
  for (int i = 1; i <= 5; i++) {
    for (int j = 1; j <= 5; j++) {
      int temp;
      cin >> temp;
      if (temp == 1) {
        a = i;
        b = j;
        return;
      }
    }
  }
}

int main() {
  int a, b;
  yoyo(a, b);
  cout << abs(3 - a) + abs(3 - b);
}