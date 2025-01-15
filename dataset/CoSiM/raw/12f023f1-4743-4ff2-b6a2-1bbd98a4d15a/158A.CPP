#include <iostream>
#include <vector>

using namespace std;

int main() {
  int n, k;
  cin >> n >> k;
  vector<int> data;

  for (int i = 0; i < n; i++) {
    int input;
    cin >> input;
    data.push_back(input);
  }

  int kth_value = data.at(k - 1), count = 0;

  for (int i = 0; i < data.size(); i++) {
    if (data.at(i) >= kth_value && data.at(i) > 0)
      count++;
  }

  cout << count << endl;

  return 0;
}