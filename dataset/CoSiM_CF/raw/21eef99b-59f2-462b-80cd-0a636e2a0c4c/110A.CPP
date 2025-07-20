#include <bits/stdc++.h>
#include <iostream>

using namespace std;

int main(void) {
  long long n;
  cin >> n;

  int totalLuckyNumber = 0;

  while (n > 0) {
    int isLucky = n % 10;
    if (isLucky == 4 || isLucky == 7) {
      totalLuckyNumber++;
    }
    n = n / 10;
  }

  if (totalLuckyNumber == 4 || totalLuckyNumber == 7) {
    cout << "YES\n";
  } else {
    cout << "NO\n";
  }
}
