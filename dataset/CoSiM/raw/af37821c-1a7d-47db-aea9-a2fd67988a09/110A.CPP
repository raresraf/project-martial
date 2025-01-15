/*
Link: https://codeforces.com/problemset/problem/110/A
@uthor: chavodon
*/

#include <iostream>

using namespace std;

int main() {
  long long int n;
  int lCount = 0;
  cin >> n;
  while (n != 0) {
    if (n - (n / 10) * 10 == 7 || n - (n / 10) * 10 == 4)
      lCount++;
    n = n / 10;
  }
  if (lCount == 7 || lCount == 4)
    cout << "YES";
  else
    cout << "NO";
}
