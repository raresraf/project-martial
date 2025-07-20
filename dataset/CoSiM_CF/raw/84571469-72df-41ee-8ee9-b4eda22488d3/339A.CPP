#include <algorithm> // sort
#include <iostream>
#include <stdlib.h> //exit
#include <vector>

using namespace std;

int main() {
  string a, b;
  cin >> a;
  vector<int> sum;
  for (int i = 0; i < a.length(); i += 2) {
    sum.insert(sum.end(), a[i]);
  }
  sort(sum.begin(), sum.end());
  for (vector<int>::iterator i = sum.begin(); i != sum.end(); i++) {
    b += char(*i);
    b.append("+");
  }
  b.pop_back();
  cout << b;
}
