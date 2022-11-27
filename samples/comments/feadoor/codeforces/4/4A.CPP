#include <iostream>

using namespace std;

/*******************************************************************************
 * Determines whether a watermelon of the given size can be split into two *
 * pieces of positive, even size. This is the case exactly when the original *
 * watermelon is of even size at least 4 *
 *******************************************************************************/
bool can_be_split(int size) { return ((size >= 4) && (size % 2 == 0)); }

/*******************************************************************************
 * Read the size of the watermelon from stdin, and decide if it can be split *
 *******************************************************************************/
int main() {
  // Read in the size of the watermelon.
  int size;
  cin >> size;

  // Output whether it can be split or not
  if (can_be_split(size)) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }

  return 0;
}