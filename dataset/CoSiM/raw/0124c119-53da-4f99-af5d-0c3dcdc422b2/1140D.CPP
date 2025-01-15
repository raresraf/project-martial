#include <bits/stdc++.h>
//#include <boost/multiprecision/cpp_int.hpp>

#define ll unsigned long long int
#define ld long double
#define pb push_back
#define pi 3.1415926536
#define mod 1e9 + 7
#define mp make_pair
#define wt while (t--)
#define fillarray for (int i = 0; i < n; i++)
#define testcases                                                              \
  ll t;                                                                        \
  cin >> t;                                                                    \
  while (t--)

using namespace std;
// using namespace boost::multiprecision;

int main() {
  int n;
  cin >> n;

  ll sum = 0;

  for (int i = 2; i < n; i++) {
    sum += i * (i + 1);
  }

  cout << sum;
}
