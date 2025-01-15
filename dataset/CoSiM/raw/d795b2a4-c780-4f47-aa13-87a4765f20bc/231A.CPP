#include <bits/stdc++.h>

using namespace std;

int cansolve(int arg1, int arg2, int arg3) {
  if (arg1 + arg2 + arg3 > 1)
    return true;
  else
    return false;
}

int main() {
  int count;
  cin >> count;
  int solve_num = 0;
  while (count-- > 0) {
    int Petya, Vasya, Tonya;
    cin >> Petya >> Vasya >> Tonya;
    if (cansolve(Petya, Vasya, Tonya))
      solve_num++;
  }
  cout << solve_num;
  return 0;
}
