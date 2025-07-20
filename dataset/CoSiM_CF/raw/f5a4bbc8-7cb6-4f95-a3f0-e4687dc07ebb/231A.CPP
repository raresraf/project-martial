#include <bits/stdc++.h>
using namespace std;
#define f first
#define s second
#define mp make_pair
#define pb push_back
typedef long long ll;
int main() {
  // freopen("input.txt", "r", stdin);
  ios_base::sync_with_stdio(false);
  int n;
  cin >> n;
  int count = 0;
  while (n--) {
    int sum = 0;
    for (int i = 0; i < 3; i++) {
      int xs;
      cin >> xs;
      sum += xs;
    }
    if (sum > 1)
      ++count;
  }
  cout << count << endl;
}