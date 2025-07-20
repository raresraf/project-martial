#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <set>
#include <utility>
#include <vector>
#define MX 100000
#define fs first
#define sec second
#define TC                                                                     \
  int T;                                                                       \
  cin >> T;                                                                    \
  while (T--)
#define loop(i, a, c) for (int i = 0; i < a; i++)
#define loop1(i, a, c) for (int i = 1; i <= a; i++)
#define pb(a) push_back(a)
#define all(a) (a).begin(), (a).end()
#define mp(x, y) make_pair((x), (y))
#define ll long long int
using namespace std;
int main() {
  int x, y;
  cin >> x >> y;
  int count = 0;
  int arr[x];
  for (int i = 0; i < x; i++)
    cin >> arr[i];
  for (int i = 0; i < x; i++) {
    if (arr[i] >= arr[y - 1] && arr[i] > 0) {
      count++;
    }
  }
  cout << count << endl;
}
