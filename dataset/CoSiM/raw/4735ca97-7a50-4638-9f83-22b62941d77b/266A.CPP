#include <bits/stdc++.h>
using namespace std;

int main() {
  int amt, i, count = 0;
  string stoneColor;
  cin >> amt >> stoneColor;
  for (i = 0; i < amt - 1; i++) {
    if (stoneColor[i] == stoneColor[i + 1])
      count++;
  }
  cout << count << endl;
  return 0;
}
