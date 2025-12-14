#include <bits/stdc++.h>
#include <iostream>

using namespace std;

int main() {
  long long int n;
  cin >> n;
  long long int count1 = 0;
  long long int count2 = 0;
  long long int count3 = 0;

  for (int i = 0; i < 3 * n; i++) {
    long long int input;
    cin >> input;
    if (i % 3 == 0)
      count1 += input;
    else if (i % 3 == 1)
      count2 += input;
    else
      count3 += input;
  }

  if (count1 == 0 && count2 == 0 && count3 == 0)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
  return 0;
}
