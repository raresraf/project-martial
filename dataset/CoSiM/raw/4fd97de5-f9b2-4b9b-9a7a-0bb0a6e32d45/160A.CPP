#include <bits/stdc++.h>
using namespace std;
int main() {
  int n;
  cin >> n;
  vector<int> s;
  int sum = 0;
  for (int i = 0; i < n; i++) {
    int temp;
    cin >> temp;
    sum = sum + temp;
    s.push_back(temp);
  }
  sort(s.begin(), s.end());
  int twin = 0;
  int count = 0;
  while (1) {

    if (twin > sum) {

      break;
    }
    n--;
    sum = sum - s[n];
    twin = twin + s[n];
    count++;
  }
  cout << count;
}