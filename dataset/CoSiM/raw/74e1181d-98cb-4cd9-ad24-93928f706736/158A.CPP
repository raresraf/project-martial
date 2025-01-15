#include <iostream>
#include <vector>

using namespace std;

int main() {
  vector<int> vec;
  int n, k;
  cin >> n >> k; // n numbers and selet numbers higher than k
  for (int m = 0; m < n; m++) {
    int a;
    cin >> a;
    vec.push_back(a);
  }
  int count = 0;
  for (int i = 0; i < vec.size(); i++) {
    if (vec[i] > 0) {
      if (count < k || vec[i] == vec[i - 1])
        count++;
      else
        break;
    }
  }
  cout << count << endl;

  return 0;
}
