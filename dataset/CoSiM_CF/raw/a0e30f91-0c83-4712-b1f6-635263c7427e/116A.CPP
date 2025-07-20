#include <bits/stdc++.h>
using namespace std;

int main(int argc, char const *argv[]) {
  int n;
  cin >> n;
  // vector<pair<int, int>> tram;
  // tram.push_back({1,2});
  // cout<<tram[0].first<<tram[0].second;
  int cap = 0;
  int temp = 0;
  int curr_val = 0;
  for (int i = 0; i < n; i++) {
    int a, b;
    cin >> a >> b;
    // tram.push_back({a,b});
    if (i == 0) {
      cap = a + b;
      curr_val = a + b;
    } else {
      curr_val = (curr_val - a) + b;
      temp = curr_val;
      if (temp > cap)
        cap = temp;
    }
  }
  cout << cap;

  return 0;
}