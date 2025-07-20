/**
 *  author: mahfuzz
 *  created: 15.11.2019
 **/

#include <bits/stdc++.h>
using namespace std;
#define debug(x) cout << '>' << #x << ':' << x << endl;
typedef long long ll;

int main(int argc, char *argv[]) {
  ios_base::sync_with_stdio(0);
  cin.tie(NULL);

  int n;
  cin >> n;
  vector<int> vec(n);
  int sum = 0;

  for (int i = 0; i < n; i++) {
    cin >> vec[i];
    sum += vec[i];
  }

  sort(vec.begin(), vec.end(), greater<int>());
  int cnt = 0;
  int s2 = 0;
  for (int i = 0; i < n; i++) {
    if (s2 > sum) {
      break;
    }
    s2 += vec[i];
    sum -= vec[i];
    cnt++;
  }
  cout << cnt << "\n";

  return 0;
}
