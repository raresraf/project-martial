#include "bits/stdc++.h"
using namespace std;
typedef long long ll;

int main() {
  ll num;
  cin >> num;

  ll count = 0;

  while (num > 0) {
    if (num % 10 == 4 || num % 10 == 7) {
      count++;
    }
    num = num / 10;
  }

  if (count == 4 || count == 7) {
    cout << "YES";
  } else
    std::cout << "NO";
  return 0;
}
