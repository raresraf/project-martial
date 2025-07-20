// problem link - https://codeforces.com/problemset/problem/231/A
// sumanth2303
// Team - 231A
// ACCEPTED
// One day three best friends Petya, Vasya and Tonya decided to form a team and
// take part in programming contest..
#include <bits/stdc++.h>
#include <iostream>
using namespace std;
int main() {
  int petya, vasya, tonya, n, score = 0;
  cin >> n;
  while (n--) {
    cin >> petya >> vasya >> tonya;
    if (petya + vasya + tonya >= 2) {
      score++;
    }
  }
  cout << score << endl;
  return 0;
}
