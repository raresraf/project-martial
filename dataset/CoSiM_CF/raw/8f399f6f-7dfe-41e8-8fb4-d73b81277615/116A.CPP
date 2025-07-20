#include <bits/stdc++.h>
using namespace std;
int main() {
  int n;
  cin >> n;
  int max = 0;
  int cur = 0;
  while (n > 0) {
    int exit;
    int enter;
    cin >> exit >> enter;
    cur = cur - exit;
    cur = cur + enter;
    if (cur > max)
      max = cur;
    n--;
  }
  cout << max;
}