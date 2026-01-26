#include <bits/stdc++.h>
#include <string>
using namespace std;

int main() {
  int t;
  cin >> t;
  int arr[t][3];
  for (int i = 0; i < t; i++)
    for (int j = 0; j < 3; j++)
      cin >> arr[i][j];
  vector<int> res;
  for (int j = 0; j < 3; j++) {
    int sum = 0;
    for (int i = 0; i < t; i++) {
      sum += arr[i][j];
    }
    res.push_back(sum);
  }
  if (res[0] == 0 && res[1] == 0 && res[2] == 0)
    cout << "YES";
  else
    cout << "NO";

  return 0;
}
