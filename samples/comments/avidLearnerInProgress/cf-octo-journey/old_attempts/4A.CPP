#include <bits/stdc++.h>
using namespace std;

int main() {

  int x;
  cin >> x;
  int flag = 0;
  for (int i = 1; i < x; i++) {
    int set = x - i;
    if (set % 2 == 0 && x % 2 == 0) {
      flag = 1;
      break;
    }
  }
  if (flag == 1)
    cout << "Yes"
         << "\n";
  else
    cout << "No"
         << "\n";
}