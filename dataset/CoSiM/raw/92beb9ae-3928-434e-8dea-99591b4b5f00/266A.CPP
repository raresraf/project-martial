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
const int N = 1e5;
static string str;
static int n;
int main() {
  IOS

      int cnt = 0;
  cin >> n;
  cin >> str;

  for (int i = 0; i < n - 1; i++)
    cnt = (str[i] == str[i + 1]) ? cnt + 1 : cnt;

  cout << cnt << endl;
  return (0);
}
