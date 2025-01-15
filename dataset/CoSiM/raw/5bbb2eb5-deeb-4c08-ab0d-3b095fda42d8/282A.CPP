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
    string xs;
    cin >> xs;
    if (xs[0] == '+' || xs[2] == '+')
      ++count;
    else
      --count;
  }
  cout << count << endl;
}