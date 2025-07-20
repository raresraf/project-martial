// Written By Mahtab Tanim
#include <bits/stdc++.h>
using namespace std;
#define MP                                                                     \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(0);                                                                  \
  cout.tie(0);
#define nl "\n"
#define End return 0
#define ll long long int
#define mp make_pair
#define pb push_back
#define ull unsigned long long int
#define pi acos(-1)
#define forn(i, a, b) for (int i = a; i <= b; i++)
int main() {
  int n, j, k;
  cin >> n >> k;
  vector<int> x;
  int count = 0;
  for (int i = 0; i < n; i++) {
    cin >> j;
    x.push_back(j);
  }
  int temp = x[k - 1];
  for (int i = 0; i < n; i++) {
    if (x[i] > 0 && x[i] >= temp)
      count++;
  }
  cout << count << endl;
  End;
}
