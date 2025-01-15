#include <bits/stdc++.h>
using namespace std;
#define f first
#define s second
#define mp make_pair
#define pb push_back
typedef long long ll;
int main(void) {
  // freopen("input.txt", "r", stdin);
  ios_base::sync_with_stdio(false);
  int n, k;
  cin >> n >> k;
  int ary[n];
  int count = 0;
  for (int i = 0; i < n; i++)
    cin >> ary[i];
  int pivot = ary[k - 1];
  for (int i = 0; i < n; i++) {
    if (ary[i] >= pivot && ary[i] > 0)
      ++count;
  }
  cout << count << endl;
}