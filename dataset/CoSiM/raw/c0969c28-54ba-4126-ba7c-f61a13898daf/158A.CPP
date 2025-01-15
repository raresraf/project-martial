#include <bits/stdc++.h>
using namespace std;

int main(int argc, char const *argv[]) {
  int n, k, temp;
  cin >> n >> k;

  int count = 0;
  int kth = 1;
  for (int i = 1; i <= n; ++i) {
    cin >> temp;
    if (i == k)
      kth = max(kth, temp);

    count += (temp >= kth);
  }
  cout << count << endl;
  return 0;
}