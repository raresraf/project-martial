#include <iostream>
#include <vector>

using namespace std;

int main() {
  int n, k, count, i;
  int input;
  cin >> n >> k;
  k = k - 1; // Index starts from 0

  vector<int> arr;

  for (int i = 0; i < n; i++) {
    cin >> input;
    arr.push_back(input);
  }

  if (arr[k] > 0) {
    count = i = k;
    count = count + 1;
    while (arr[k] == arr[++i]) // Because decreasing array list we count after
                               // kth finisher's score
    {
      count++;
    }
    cout << count;
  } else {
    count = i = 0;

    while (arr[i++] > 0) // Counting up before zeroth score
    {
      count++;
    }
    cout << count;
  }
  return 0;
}
