#include <iostream>
using namespace std;
#define ll long long

int main() {
  ll t, n, p, c, sum;

  cin >> t;

  while (t--) {
    sum = 0;
    cin >> n;
    cin >> p;
    n--;
    while (n--) {
      cin >> c;
      if (p * c > 0) { // check is it in same sign or not
        if (c > p)
          p = c;
      } else {
        sum += p; // add the max from same sign elements
        p = c;
      }
    }
    cout << sum + p << endl; // add max element from last sub-array of same sign
  }

  return 0;
}