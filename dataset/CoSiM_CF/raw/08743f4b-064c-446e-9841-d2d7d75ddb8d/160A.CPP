#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>

using namespace std;

int main() {

#ifndef ONLINE_JUDGE
  freopen("c.in", "rt", stdin);
#endif // ONLINE_JUDGE

  int t, input, sum = 0, counter = 0, sum2 = 0;
  vector<int> arr;

  cin >> t;

  while (t--) {
    cin >> input;
    arr.push_back(input);
  }

  for (int i = 0; i < arr.size(); i++) {
    sum += arr[i];
  }

  sort(arr.rbegin(), arr.rend());

  sum /= 2;
  sum += 1;

  for (int i = 0; i < arr.size(); i++) {
    sum2 += arr[i];
    counter++;
    if (sum2 >= sum)
      break;
  }

  cout << counter;

  return 0;
}
