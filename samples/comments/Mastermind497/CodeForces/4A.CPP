#include <iostream>
using namespace std;

int W; // weight of the watermelon

int main() {
  cin >> W;
  if (W % 2 == 0 && W > 3)
    cout << "YES";
  else
    cout << "NO";
}
