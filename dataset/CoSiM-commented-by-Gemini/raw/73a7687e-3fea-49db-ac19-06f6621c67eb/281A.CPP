/**
* Git >> Mohsin_Riad


















**/
#include <bits/stdc++.h>
using namespace std;

#define IOS                                                                    \
  ios_base::sync_with_stdio(0);                                                \
  cin.tie(0);                                                                  \
  cout.tie(0);
#define mx 1000
#define endl "\n"
#define pb push_back
#define pob pop_back
typedef long long ll;
typedef double dd;
const int N = 1e6;

int main() {
  IOS

      char str[1005];
  cin >> str;

  if (str[0] >= 97 && 122 >= str[0])
    str[0] -= 32;

  cout << str << endl;
  return (0);
}
