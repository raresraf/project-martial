/**
 * @file 158A.CPP
 * @brief An incorrect attempt to solve the "Next Round" problem.
 *
 * This program tries to count the number of participants who advance to the next
 * round. The rule is that a participant advances if they have a positive score
 * and their score is at least that of the k-th place finisher.
 *
 * @b WARNING: The logic in this implementation is flawed and does not correctly
 *    solve the problem. It makes decisions by comparing a participant's score
 *    only to the immediately preceding one, not to the global k-th place score.
 *    It also has a bug in checking for positive scores.
 *
 * @b Algorithm: The code iterates through the scores sequentially, incrementing a
 *    counter. It attempts to break out of the loop based on local comparisons,
 *    which is not a valid strategy for this problem.
 * @b Time_Complexity: O(N), where N is the number of participants.
 * @b Space_Complexity: O(1).
 */
#include <iostream>
using namespace std;

int main() {
  int a, b, cont = 0; // a: num participants (N), b: cutoff rank (k), cont: counter
  cin >> a >> b;
  int aux, aux2; // aux: previous score, aux2: current score

  // Loop through each participant's score as it's read.
  for (int i = 0; i < a; i++) {
    aux = aux2;
    cin >> aux2;

    // BUG: This condition is `(!aux2) > 0`, which is only true if `aux2` is 0.
    // It is likely intended to be `if (aux2 <= 0)`.
    if (!aux2 > 0)
      break;

    // FLAW: This logic is incorrect. It compares the current score `aux2`
    // with the previous score `aux`. The correct logic requires comparing against
    // the score of the b-th participant, which requires storing at least the
    // first `b` scores.
    if ((i + 1) > b && aux2 < aux)
      break;
    
    cont++;
  }
  cout << cont << endl;
  return 0;
}