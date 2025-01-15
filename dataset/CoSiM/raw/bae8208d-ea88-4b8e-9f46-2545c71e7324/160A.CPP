#include <bits/stdc++.h>
#include <iostream>
#include <math.h>

using namespace std;
vector<int> arr;
int main() {
  int n;
  int total = 0;
  int val;
  int takenTotal;
  while (cin >> n) {
    total = 0;
    for (int i = 0; i < n; i++) {
      cin >> val;
      arr.push_back(val);
      total = total + val;
    }
    int i = arr.size();
    sort(arr.begin(), arr.begin() + i);
    takenTotal = 0;
    do {
      takenTotal = takenTotal + arr[--i];
    } while ((total - takenTotal) >= takenTotal && (i - 1 > -1));
    cout << arr.size() - i << endl;
  }
}
