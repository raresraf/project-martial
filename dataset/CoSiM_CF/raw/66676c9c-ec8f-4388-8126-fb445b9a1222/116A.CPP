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
  int xs = 0;
  int current = 0;
  while (n--) {
    int x, y;
    cin >> x >> y;
    int total = y - x;
    current += total;
    if (current > xs)
      xs = current;
  }
  cout << xs << endl;
}